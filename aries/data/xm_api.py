"""
XM API Data Collector

Handles data collection from the Colombian XM energy market API
using the pydataxm library for historical and real-time data.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import logging
import requests
import time

try:
    from pydataxm import *
except ImportError:
    print("Warning: pydataxm not installed. Install with: pip install pydataxm")
    # Create mock classes for development
    class ReadDB:
        def __init__(self, *args, **kwargs):
            pass
        def request_data(self, *args, **kwargs):
            return pd.DataFrame()
    
    class ReadDB_simem:
        def __init__(self, *args, **kwargs):
            pass
        def request_data(self, *args, **kwargs):
            return pd.DataFrame()

logger = logging.getLogger(__name__)


class XMDataCollector:
    """
    Data collector for XM (Colombian Energy Market) API.
    
    Provides access to historical and real-time energy market data
    including prices, demand, supply, and other market indicators.
    """
    
    def __init__(self):
        """Initialize the XM data collector."""
        self.base_url = "https://www.xm.com.co"
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Aries-Trading-Agent/1.0'
        })
        
        # Initialize pydataxm readers
        try:
            self.db_reader = ReadDB()
            self.simem_reader = ReadDB_simem()
        except Exception as e:
            logger.warning(f"Could not initialize pydataxm readers: {e}")
            self.db_reader = None
            self.simem_reader = None
    
    def get_historical_prices(self, start_date: datetime, end_date: datetime) -> pd.DataFrame:
        """
        Get historical energy prices from XM.
        
        Args:
            start_date: Start date for data collection
            end_date: End date for data collection
            
        Returns:
            DataFrame with historical price data
        """
        logger.info(f"Fetching XM historical prices from {start_date} to {end_date}")
        
        try:
            if self.db_reader is None:
                logger.error("pydataxm not available, returning empty DataFrame")
                return pd.DataFrame()
            
            # Format dates for XM API
            start_str = start_date.strftime('%Y-%m-%d')
            end_str = end_date.strftime('%Y-%m-%d')
            
            # Get price data from XM
            price_data = self.db_reader.request_data(
                'PrecioEnergia',
                start_str,
                end_str
            )
            
            if price_data.empty:
                logger.warning("No price data received from XM API")
                return pd.DataFrame()
            
            # Process and clean the data
            processed_data = self._process_price_data(price_data)
            
            logger.info(f"Retrieved {len(processed_data)} price records")
            return processed_data
            
        except Exception as e:
            logger.error(f"Error fetching XM historical prices: {e}")
            return pd.DataFrame()
    
    def get_historical_demand(self, start_date: datetime, end_date: datetime) -> pd.DataFrame:
        """
        Get historical demand data from XM.
        
        Args:
            start_date: Start date for data collection
            end_date: End date for data collection
            
        Returns:
            DataFrame with historical demand data
        """
        logger.info(f"Fetching XM historical demand from {start_date} to {end_date}")
        
        try:
            if self.db_reader is None:
                logger.error("pydataxm not available, returning empty DataFrame")
                return pd.DataFrame()
            
            start_str = start_date.strftime('%Y-%m-%d')
            end_str = end_date.strftime('%Y-%m-%d')
            
            demand_data = self.db_reader.request_data(
                'Demanda',
                start_str,
                end_str
            )
            
            if demand_data.empty:
                logger.warning("No demand data received from XM API")
                return pd.DataFrame()
            
            processed_data = self._process_demand_data(demand_data)
            logger.info(f"Retrieved {len(processed_data)} demand records")
            return processed_data
            
        except Exception as e:
            logger.error(f"Error fetching XM historical demand: {e}")
            return pd.DataFrame()
    
    def get_historical_supply(self, start_date: datetime, end_date: datetime) -> pd.DataFrame:
        """
        Get historical supply data from XM.
        
        Args:
            start_date: Start date for data collection
            end_date: End date for data collection
            
        Returns:
            DataFrame with historical supply data
        """
        logger.info(f"Fetching XM historical supply from {start_date} to {end_date}")
        
        try:
            if self.db_reader is None:
                logger.error("pydataxm not available, returning empty DataFrame")
                return pd.DataFrame()
            
            start_str = start_date.strftime('%Y-%m-%d')
            end_str = end_date.strftime('%Y-%m-%d')
            
            supply_data = self.db_reader.request_data(
                'Oferta',
                start_str,
                end_str
            )
            
            if supply_data.empty:
                logger.warning("No supply data received from XM API")
                return pd.DataFrame()
            
            processed_data = self._process_supply_data(supply_data)
            logger.info(f"Retrieved {len(processed_data)} supply records")
            return processed_data
            
        except Exception as e:
            logger.error(f"Error fetching XM historical supply: {e}")
            return pd.DataFrame()
    
    def get_weather_data(self, start_date: datetime, end_date: datetime) -> pd.DataFrame:
        """
        Get weather data that might affect energy prices.
        
        Args:
            start_date: Start date for data collection
            end_date: End date for data collection
            
        Returns:
            DataFrame with weather data
        """
        logger.info(f"Fetching weather data from {start_date} to {end_date}")
        
        try:
            if self.simem_reader is None:
                logger.error("pydataxm SIMEM not available, returning empty DataFrame")
                return pd.DataFrame()
            
            start_str = start_date.strftime('%Y-%m-%d')
            end_str = end_date.strftime('%Y-%m-%d')
            
            # Get weather data from SIMEM
            weather_data = self.simem_reader.request_data(
                'Temperatura',
                start_str,
                end_str
            )
            
            if weather_data.empty:
                logger.warning("No weather data received from SIMEM API")
                return pd.DataFrame()
            
            processed_data = self._process_weather_data(weather_data)
            logger.info(f"Retrieved {len(processed_data)} weather records")
            return processed_data
            
        except Exception as e:
            logger.error(f"Error fetching weather data: {e}")
            return pd.DataFrame()
    
    def get_real_time_data(self) -> Dict:
        """
        Get real-time market data from XM.
        
        Returns:
            Dictionary with current market data
        """
        logger.info("Fetching real-time XM data")
        
        try:
            # This would typically involve real-time API calls
            # For now, we'll return a mock structure
            current_time = datetime.now()
            
            return {
                'timestamp': current_time,
                'current_price': None,  # Would be fetched from real-time API
                'current_demand': None,
                'current_supply': None,
                'market_status': 'OPEN'  # OPEN, CLOSED, etc.
            }
            
        except Exception as e:
            logger.error(f"Error fetching real-time data: {e}")
            return {}
    
    def _process_price_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """Process and clean price data from XM API."""
        if data.empty:
            return data
        
        # Standardize column names
        processed = data.copy()
        
        # Ensure timestamp column exists and is properly formatted
        if 'Fecha' in processed.columns:
            processed['timestamp'] = pd.to_datetime(processed['Fecha'])
        elif 'Date' in processed.columns:
            processed['timestamp'] = pd.to_datetime(processed['Date'])
        
        # Standardize price column
        price_columns = ['Precio', 'Price', 'Valor', 'PrecioEnergia']
        for col in price_columns:
            if col in processed.columns:
                processed['price'] = pd.to_numeric(processed[col], errors='coerce')
                break
        
        # Clean and validate data
        processed = processed.dropna(subset=['timestamp', 'price'])
        processed = processed.sort_values('timestamp')
        
        return processed[['timestamp', 'price']]
    
    def _process_demand_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """Process and clean demand data from XM API."""
        if data.empty:
            return data
        
        processed = data.copy()
        
        # Standardize timestamp
        if 'Fecha' in processed.columns:
            processed['timestamp'] = pd.to_datetime(processed['Fecha'])
        
        # Standardize demand column
        demand_columns = ['Demanda', 'Demand', 'Consumo', 'ConsumoEnergia']
        for col in demand_columns:
            if col in processed.columns:
                processed['demand'] = pd.to_numeric(processed[col], errors='coerce')
                break
        
        processed = processed.dropna(subset=['timestamp', 'demand'])
        processed = processed.sort_values('timestamp')
        
        return processed[['timestamp', 'demand']]
    
    def _process_supply_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """Process and clean supply data from XM API."""
        if data.empty:
            return data
        
        processed = data.copy()
        
        # Standardize timestamp
        if 'Fecha' in processed.columns:
            processed['timestamp'] = pd.to_datetime(processed['Fecha'])
        
        # Standardize supply column
        supply_columns = ['Oferta', 'Supply', 'Generacion', 'GeneracionEnergia']
        for col in supply_columns:
            if col in processed.columns:
                processed['supply'] = pd.to_numeric(processed[col], errors='coerce')
                break
        
        processed = processed.dropna(subset=['timestamp', 'supply'])
        processed = processed.sort_values('timestamp')
        
        return processed[['timestamp', 'supply']]
    
    def _process_weather_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """Process and clean weather data from SIMEM API."""
        if data.empty:
            return data
        
        processed = data.copy()
        
        # Standardize timestamp
        if 'Fecha' in processed.columns:
            processed['timestamp'] = pd.to_datetime(processed['Fecha'])
        
        # Standardize weather columns
        if 'Temperatura' in processed.columns:
            processed['temperature'] = pd.to_numeric(processed['Temperatura'], errors='coerce')
        
        if 'Humedad' in processed.columns:
            processed['humidity'] = pd.to_numeric(processed['Humedad'], errors='coerce')
        
        if 'VelocidadViento' in processed.columns:
            processed['wind_speed'] = pd.to_numeric(processed['VelocidadViento'], errors='coerce')
        
        processed = processed.dropna(subset=['timestamp'])
        processed = processed.sort_values('timestamp')
        
        return processed
    
    def get_market_indicators(self, start_date: datetime, end_date: datetime) -> pd.DataFrame:
        """
        Get comprehensive market indicators combining prices, demand, and supply.
        
        Args:
            start_date: Start date for data collection
            end_date: End date for data collection
            
        Returns:
            DataFrame with comprehensive market data
        """
        logger.info("Fetching comprehensive market indicators")
        
        # Get all data types
        prices = self.get_historical_prices(start_date, end_date)
        demand = self.get_historical_demand(start_date, end_date)
        supply = self.get_historical_supply(start_date, end_date)
        weather = self.get_weather_data(start_date, end_date)
        
        # Combine all data
        all_data = []
        
        if not prices.empty:
            all_data.append(prices)
        if not demand.empty:
            all_data.append(demand)
        if not supply.empty:
            all_data.append(supply)
        if not weather.empty:
            all_data.append(weather)
        
        if not all_data:
            logger.warning("No market data available")
            return pd.DataFrame()
        
        # Merge all data on timestamp
        combined = all_data[0]
        for data in all_data[1:]:
            combined = combined.merge(data, on='timestamp', how='outer')
        
        # Fill missing values and clean
        combined = combined.sort_values('timestamp')
        combined = combined.fillna(method='ffill').fillna(method='bfill')
        
        # Add derived features
        combined = self._add_derived_features(combined)
        
        logger.info(f"Combined market data: {len(combined)} records")
        return combined
    
    def _add_derived_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Add derived features to market data."""
        if data.empty:
            return data
        
        df = data.copy()
        
        # Add time-based features
        df['hour'] = df['timestamp'].dt.hour
        df['day_of_week'] = df['timestamp'].dt.dayofweek
        df['month'] = df['timestamp'].dt.month
        df['is_weekend'] = df['day_of_week'].isin([5, 6])
        
        # Add price-based features
        if 'price' in df.columns:
            df['price_change'] = df['price'].pct_change()
            df['price_ma_24h'] = df['price'].rolling(window=24, min_periods=1).mean()
            df['price_volatility'] = df['price'].rolling(window=24, min_periods=1).std()
        
        # Add demand-supply balance
        if 'demand' in df.columns and 'supply' in df.columns:
            df['demand_supply_ratio'] = df['demand'] / df['supply']
            df['excess_supply'] = df['supply'] - df['demand']
        
        return df
