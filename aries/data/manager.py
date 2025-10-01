"""
Data Manager for Aries Trading Agent

Centralized data management system that coordinates data collection,
preprocessing, and storage for the energy market trading agent.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Union
import logging
from pathlib import Path
import sqlite3
import json

from .xm_api import XMDataCollector
from .san_andres import SanAndresDataCollector
from .preprocessor import DataPreprocessor

logger = logging.getLogger(__name__)


class DataManager:
    """
    Centralized data management system for the Aries trading agent.
    
    Handles data collection from multiple sources, preprocessing,
    and provides unified access to market data.
    """
    
    def __init__(self, config: Dict = None):
        """
        Initialize the DataManager.
        
        Args:
            config: Configuration dictionary with data sources and parameters
        """
        self.config = config or self._default_config()
        self.db_path = Path(self.config.get('database_path', 'data/market_data.db'))
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Initialize data collectors
        self.xm_collector = XMDataCollector()
        self.san_andres_collector = SanAndresDataCollector()
        self.preprocessor = DataPreprocessor()
        
        # Initialize database
        self._init_database()
        
    def _default_config(self) -> Dict:
        """Return default configuration."""
        return {
            'database_path': 'data/market_data.db',
            'update_frequency': 3600,  # seconds
            'max_history_days': 365,
            'sources': {
                'xm': {
                    'enabled': True,
                    'update_interval': 3600
                },
                'san_andres': {
                    'enabled': True,
                    'update_interval': 1800
                }
            }
        }
    
    def _init_database(self):
        """Initialize SQLite database for data storage."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute('''
                CREATE TABLE IF NOT EXISTS market_prices (
                    timestamp TEXT PRIMARY KEY,
                    price REAL,
                    volume REAL,
                    demand REAL,
                    supply REAL,
                    source TEXT
                )
            ''')
            
            conn.execute('''
                CREATE TABLE IF NOT EXISTS weather_data (
                    timestamp TEXT PRIMARY KEY,
                    temperature REAL,
                    humidity REAL,
                    wind_speed REAL,
                    solar_radiation REAL,
                    location TEXT
                )
            ''')
            
            conn.execute('''
                CREATE TABLE IF NOT EXISTS san_andres_data (
                    timestamp TEXT PRIMARY KEY,
                    consumption REAL,
                    generation REAL,
                    storage_level REAL,
                    diesel_price REAL
                )
            ''')
            
            conn.commit()
    
    def collect_market_data(self, start_date: datetime, end_date: datetime) -> pd.DataFrame:
        """
        Collect market data from all sources for the specified date range.
        
        Args:
            start_date: Start date for data collection
            end_date: End date for data collection
            
        Returns:
            DataFrame with market data
        """
        logger.info(f"Collecting market data from {start_date} to {end_date}")
        
        all_data = []
        
        # Collect XM data
        if self.config['sources']['xm']['enabled']:
            try:
                xm_data = self.xm_collector.get_historical_prices(start_date, end_date)
                if not xm_data.empty:
                    xm_data['source'] = 'XM'
                    all_data.append(xm_data)
                    logger.info(f"Collected {len(xm_data)} XM records")
            except Exception as e:
                logger.error(f"Error collecting XM data: {e}")
        
        # Collect San Andrés data
        if self.config['sources']['san_andres']['enabled']:
            try:
                sa_data = self.san_andres_collector.get_historical_data(start_date, end_date)
                if not sa_data.empty:
                    sa_data['source'] = 'SAN_ANDRES'
                    all_data.append(sa_data)
                    logger.info(f"Collected {len(sa_data)} San Andrés records")
            except Exception as e:
                logger.error(f"Error collecting San Andrés data: {e}")
        
        if not all_data:
            logger.warning("No data collected from any source")
            return pd.DataFrame()
        
        # Combine and preprocess data
        combined_data = pd.concat(all_data, ignore_index=True)
        processed_data = self.preprocessor.process_market_data(combined_data)
        
        # Store in database
        self._store_market_data(processed_data)
        
        return processed_data
    
    def get_latest_data(self, hours: int = 24) -> pd.DataFrame:
        """
        Get the latest market data.
        
        Args:
            hours: Number of hours of data to retrieve
            
        Returns:
            DataFrame with latest market data
        """
        end_time = datetime.now()
        start_time = end_time - timedelta(hours=hours)
        
        return self.collect_market_data(start_time, end_time)
    
    def get_historical_data(self, days: int = 30) -> pd.DataFrame:
        """
        Get historical market data.
        
        Args:
            days: Number of days of historical data to retrieve
            
        Returns:
            DataFrame with historical market data
        """
        end_time = datetime.now()
        start_time = end_time - timedelta(days=days)
        
        return self.collect_market_data(start_time, end_time)
    
    def _store_market_data(self, data: pd.DataFrame):
        """Store market data in the database."""
        if data.empty:
            return
        
        with sqlite3.connect(self.db_path) as conn:
            for _, row in data.iterrows():
                conn.execute('''
                    INSERT OR REPLACE INTO market_prices 
                    (timestamp, price, volume, demand, supply, source)
                    VALUES (?, ?, ?, ?, ?, ?)
                ''', (
                    row['timestamp'].isoformat() if hasattr(row['timestamp'], 'isoformat') else str(row['timestamp']),
                    row.get('price', None),
                    row.get('volume', None),
                    row.get('demand', None),
                    row.get('supply', None),
                    row.get('source', 'UNKNOWN')
                ))
            conn.commit()
    
    def get_stored_data(self, start_date: datetime = None, end_date: datetime = None) -> pd.DataFrame:
        """
        Retrieve stored data from the database.
        
        Args:
            start_date: Start date for data retrieval
            end_date: End date for data retrieval
            
        Returns:
            DataFrame with stored market data
        """
        with sqlite3.connect(self.db_path) as conn:
            query = "SELECT * FROM market_prices"
            params = []
            
            if start_date:
                query += " WHERE timestamp >= ?"
                params.append(start_date.isoformat())
            
            if end_date:
                if start_date:
                    query += " AND timestamp <= ?"
                else:
                    query += " WHERE timestamp <= ?"
                params.append(end_date.isoformat())
            
            query += " ORDER BY timestamp"
            
            df = pd.read_sql_query(query, conn, params=params)
            
            if not df.empty:
                df['timestamp'] = pd.to_datetime(df['timestamp'])
                df = df.set_index('timestamp')
            
            return df
    
    def update_data(self):
        """Update data from all sources."""
        logger.info("Starting data update")
        
        # Get last update time
        last_update = self._get_last_update_time()
        update_interval = timedelta(seconds=self.config['update_frequency'])
        
        if datetime.now() - last_update < update_interval:
            logger.info("Data is up to date, skipping update")
            return
        
        # Update data
        end_time = datetime.now()
        start_time = last_update
        
        self.collect_market_data(start_time, end_time)
        self._update_last_update_time()
        
        logger.info("Data update completed")
    
    def _get_last_update_time(self) -> datetime:
        """Get the last data update time."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.execute("SELECT MAX(timestamp) FROM market_prices")
                result = cursor.fetchone()
                if result[0]:
                    return datetime.fromisoformat(result[0])
        except Exception as e:
            logger.error(f"Error getting last update time: {e}")
        
        # Default to 24 hours ago if no data
        return datetime.now() - timedelta(hours=24)
    
    def _update_last_update_time(self):
        """Update the last update time."""
        # This could be stored in a separate metadata table
        # For now, we'll use the latest timestamp in market_prices
        pass
    
    def get_data_summary(self) -> Dict:
        """
        Get a summary of available data.
        
        Returns:
            Dictionary with data summary statistics
        """
        with sqlite3.connect(self.db_path) as conn:
            # Get data counts by source
            cursor = conn.execute("""
                SELECT source, COUNT(*) as count, 
                       MIN(timestamp) as earliest, 
                       MAX(timestamp) as latest
                FROM market_prices 
                GROUP BY source
            """)
            
            sources = {}
            for row in cursor.fetchall():
                sources[row[0]] = {
                    'count': row[1],
                    'earliest': row[2],
                    'latest': row[3]
                }
            
            # Get price statistics
            cursor = conn.execute("""
                SELECT AVG(price) as avg_price, 
                       MIN(price) as min_price, 
                       MAX(price) as max_price,
                       COUNT(DISTINCT DATE(timestamp)) as days
                FROM market_prices
            """)
            
            stats = cursor.fetchone()
            
            return {
                'sources': sources,
                'price_stats': {
                    'average': stats[0],
                    'minimum': stats[1],
                    'maximum': stats[2],
                    'days_covered': stats[3]
                },
                'last_update': self._get_last_update_time().isoformat()
            }
