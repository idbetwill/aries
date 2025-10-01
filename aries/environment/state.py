"""
Market State Management for Energy Trading Environment

Handles the representation and management of market state
for the reinforcement learning environment.
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
import logging

logger = logging.getLogger(__name__)


class MarketState:
    """
    Manages the state representation for the energy market environment.
    
    Handles state encoding, normalization, and feature engineering
    for the reinforcement learning agent.
    """
    
    def __init__(self, config: Dict = None):
        """
        Initialize the MarketState manager.
        
        Args:
            config: Configuration dictionary for state management
        """
        self.config = config or self._default_config()
        self.state_history = []
        self.normalization_stats = {}
        
    def _default_config(self) -> Dict:
        """Return default state configuration."""
        return {
            'state_features': [
                'price', 'price_change', 'price_volatility_24h',
                'demand', 'supply', 'demand_supply_ratio',
                'hour', 'day_of_week', 'is_weekend',
                'position', 'capital', 'portfolio_value'
            ],
            'normalize_features': True,
            'include_forecasts': True,
            'forecast_horizon': 24,
            'state_history_length': 10
        }
    
    def encode_state(self, market_data: Dict, agent_state: Dict) -> np.ndarray:
        """
        Encode market and agent state into a feature vector.
        
        Args:
            market_data: Current market data dictionary
            agent_state: Current agent state dictionary
            
        Returns:
            Encoded state vector
        """
        state_vector = []
        
        for feature in self.config['state_features']:
            if feature in market_data:
                value = market_data[feature]
            elif feature in agent_state:
                value = agent_state[feature]
            else:
                value = 0.0
            
            # Handle NaN values
            if pd.isna(value) or value is None:
                value = 0.0
            
            # Normalize if configured
            if self.config['normalize_features']:
                value = self._normalize_feature(feature, value)
            
            state_vector.append(float(value))
        
        # Add forecast features if configured
        if self.config['include_forecasts'] and 'forecasts' in market_data:
            forecast_vector = self._encode_forecasts(market_data['forecasts'])
            state_vector.extend(forecast_vector)
        
        return np.array(state_vector, dtype=np.float32)
    
    def _normalize_feature(self, feature_name: str, value: float) -> float:
        """
        Normalize a feature value.
        
        Args:
            feature_name: Name of the feature
            value: Raw feature value
            
        Returns:
            Normalized feature value
        """
        # Get normalization statistics
        if feature_name not in self.normalization_stats:
            return value
        
        stats = self.normalization_stats[feature_name]
        mean = stats.get('mean', 0.0)
        std = stats.get('std', 1.0)
        
        if std == 0:
            return 0.0
        
        return (value - mean) / std
    
    def _encode_forecasts(self, forecasts: Dict) -> List[float]:
        """
        Encode forecast data into the state vector.
        
        Args:
            forecasts: Dictionary containing forecast data
            
        Returns:
            List of forecast features
        """
        forecast_vector = []
        
        # Price forecasts
        if 'price_forecast' in forecasts:
            price_forecast = forecasts['price_forecast']
            if isinstance(price_forecast, (list, np.ndarray)):
                # Take first few forecast points
                forecast_vector.extend(price_forecast[:5])
            else:
                forecast_vector.append(float(price_forecast))
        
        # Demand forecasts
        if 'demand_forecast' in forecasts:
            demand_forecast = forecasts['demand_forecast']
            if isinstance(demand_forecast, (list, np.ndarray)):
                forecast_vector.extend(demand_forecast[:3])
            else:
                forecast_vector.append(float(demand_forecast))
        
        # Supply forecasts
        if 'supply_forecast' in forecasts:
            supply_forecast = forecasts['supply_forecast']
            if isinstance(supply_forecast, (list, np.ndarray)):
                forecast_vector.extend(supply_forecast[:3])
            else:
                forecast_vector.append(float(supply_forecast))
        
        # Forecast uncertainty
        if 'forecast_uncertainty' in forecasts:
            uncertainty = forecasts['forecast_uncertainty']
            if isinstance(uncertainty, (list, np.ndarray)):
                forecast_vector.extend(uncertainty[:3])
            else:
                forecast_vector.append(float(uncertainty))
        
        return forecast_vector
    
    def update_normalization_stats(self, data: pd.DataFrame):
        """
        Update normalization statistics from historical data.
        
        Args:
            data: Historical data DataFrame
        """
        logger.info("Updating normalization statistics")
        
        for feature in self.config['state_features']:
            if feature in data.columns:
                values = data[feature].dropna()
                if len(values) > 0:
                    self.normalization_stats[feature] = {
                        'mean': values.mean(),
                        'std': values.std(),
                        'min': values.min(),
                        'max': values.max()
                    }
    
    def get_state_size(self) -> int:
        """Get the size of the state vector."""
        base_size = len(self.config['state_features'])
        
        if self.config['include_forecasts']:
            # Add forecast features (estimated)
            forecast_size = 15  # Price (5) + Demand (3) + Supply (3) + Uncertainty (3) + Other (1)
            base_size += forecast_size
        
        return base_size
    
    def get_feature_names(self) -> List[str]:
        """Get the names of all features in the state vector."""
        feature_names = self.config['state_features'].copy()
        
        if self.config['include_forecasts']:
            forecast_features = [
                'price_forecast_1h', 'price_forecast_2h', 'price_forecast_3h', 
                'price_forecast_6h', 'price_forecast_12h',
                'demand_forecast_1h', 'demand_forecast_6h', 'demand_forecast_12h',
                'supply_forecast_1h', 'supply_forecast_6h', 'supply_forecast_12h',
                'uncertainty_1h', 'uncertainty_6h', 'uncertainty_12h',
                'forecast_confidence'
            ]
            feature_names.extend(forecast_features)
        
        return feature_names
    
    def add_state_to_history(self, state: np.ndarray):
        """
        Add state to history for temporal features.
        
        Args:
            state: State vector to add
        """
        self.state_history.append(state.copy())
        
        # Keep only recent history
        max_length = self.config['state_history_length']
        if len(self.state_history) > max_length:
            self.state_history = self.state_history[-max_length:]
    
    def get_temporal_features(self) -> np.ndarray:
        """
        Get temporal features from state history.
        
        Returns:
            Temporal feature vector
        """
        if len(self.state_history) < 2:
            return np.array([])
        
        # Calculate temporal features
        recent_states = np.array(self.state_history[-5:])  # Last 5 states
        
        # State changes
        state_changes = np.diff(recent_states, axis=0)
        
        # Trends (slopes)
        trends = []
        for i in range(len(recent_states[0])):
            if len(recent_states) > 1:
                trend = np.polyfit(range(len(recent_states)), recent_states[:, i], 1)[0]
                trends.append(trend)
            else:
                trends.append(0.0)
        
        # Volatility (standard deviation of changes)
        volatility = np.std(state_changes, axis=0) if len(state_changes) > 0 else np.zeros(len(recent_states[0]))
        
        # Combine temporal features
        temporal_features = np.concatenate([
            state_changes[-1] if len(state_changes) > 0 else np.zeros(len(recent_states[0])),
            trends,
            volatility
        ])
        
        return temporal_features
    
    def get_market_regime(self, market_data: Dict) -> str:
        """
        Determine the current market regime.
        
        Args:
            market_data: Current market data
            
        Returns:
            Market regime string
        """
        price = market_data.get('price', 0)
        volatility = market_data.get('price_volatility_24h', 0)
        demand_supply_ratio = market_data.get('demand_supply_ratio', 1)
        
        # High volatility regime
        if volatility > 0.1:
            return 'high_volatility'
        
        # High demand regime
        elif demand_supply_ratio > 1.2:
            return 'high_demand'
        
        # Low demand regime
        elif demand_supply_ratio < 0.8:
            return 'low_demand'
        
        # Normal regime
        else:
            return 'normal'
    
    def get_state_importance(self, state: np.ndarray) -> Dict[str, float]:
        """
        Calculate the importance of different state components.
        
        Args:
            state: State vector
            
        Returns:
            Dictionary with feature importance scores
        """
        feature_names = self.get_feature_names()
        importance = {}
        
        for i, feature in enumerate(feature_names):
            if i < len(state):
                # Simple importance based on absolute value
                importance[feature] = abs(float(state[i]))
        
        return importance
    
    def get_state_summary(self, state: np.ndarray) -> Dict:
        """
        Get a summary of the current state.
        
        Args:
            state: State vector
            
        Returns:
            State summary dictionary
        """
        feature_names = self.get_feature_names()
        
        summary = {
            'state_size': len(state),
            'non_zero_features': np.count_nonzero(state),
            'max_value': float(np.max(state)),
            'min_value': float(np.min(state)),
            'mean_value': float(np.mean(state)),
            'std_value': float(np.std(state))
        }
        
        # Add feature-specific information
        for i, feature in enumerate(feature_names):
            if i < len(state):
                summary[f'{feature}_value'] = float(state[i])
        
        return summary
