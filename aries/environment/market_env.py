"""
Energy Market Environment for Reinforcement Learning

A Gymnasium-based environment that simulates the Colombian energy market
for training risk-averse trading agents.
"""

import gymnasium as gym
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any, Union
import logging
from gymnasium import spaces
import warnings

from .state import MarketState
from .rewards import RewardCalculator
from .actions import ActionSpace

warnings.filterwarnings('ignore')
logger = logging.getLogger(__name__)


class EnergyMarketEnv(gym.Env):
    """
    Energy Market Environment for Reinforcement Learning.
    
    This environment simulates the Colombian energy market (XM) with
    realistic price dynamics, demand/supply patterns, and trading mechanics.
    """
    
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 4}
    
    def __init__(self, 
                 data: pd.DataFrame = None,
                 config: Dict = None,
                 initial_capital: float = 100000.0,
                 max_position_size: float = 1000.0,
                 transaction_cost: float = 0.001,
                 risk_aversion_lambda: float = 0.5):
        """
        Initialize the Energy Market Environment.
        
        Args:
            data: Historical market data DataFrame
            config: Environment configuration dictionary
            initial_capital: Starting capital in COP
            max_position_size: Maximum position size in MWh
            transaction_cost: Transaction cost as percentage
            risk_aversion_lambda: Risk aversion coefficient
        """
        super().__init__()
        
        self.config = config or self._default_config()
        self.data = data
        self.initial_capital = initial_capital
        self.max_position_size = max_position_size
        self.transaction_cost = transaction_cost
        self.risk_aversion_lambda = risk_aversion_lambda
        
        # Initialize components
        self.state_manager = MarketState(config=self.config)
        self.reward_calculator = RewardCalculator(
            risk_aversion_lambda=risk_aversion_lambda,
            transaction_cost=transaction_cost
        )
        self.action_space_manager = ActionSpace(max_position_size=max_position_size)
        
        # Environment state
        self.current_step = 0
        self.max_steps = len(data) if data is not None else 1000
        self.episode_data = None
        self.episode_start_time = None
        
        # Trading state
        self.capital = initial_capital
        self.position = 0.0  # Current position in MWh
        self.portfolio_value = initial_capital
        self.trade_history = []
        self.price_history = []
        
        # Define action and observation spaces
        self._setup_spaces()
        
        logger.info(f"Energy Market Environment initialized with {self.max_steps} steps")
    
    def _default_config(self) -> Dict:
        """Return default environment configuration."""
        return {
            'state_features': [
                'price', 'price_change', 'price_volatility_24h',
                'demand', 'supply', 'demand_supply_ratio',
                'hour', 'day_of_week', 'is_weekend',
                'position', 'capital', 'portfolio_value'
            ],
            'forecast_horizon': 24,  # hours
            'trading_frequency': 1,  # hours
            'price_impact': 0.001,  # Price impact of large trades
            'market_volatility': 0.02,  # Base market volatility
            'demand_elasticity': 0.5,  # Price elasticity of demand
            'supply_elasticity': 0.3,  # Price elasticity of supply
        }
    
    def _setup_spaces(self):
        """Setup action and observation spaces."""
        # Action space: [buy_amount, sell_amount, hold_probability]
        # Actions are continuous in [0, 1] representing fraction of max position
        self.action_space = spaces.Box(
            low=np.array([0.0, 0.0, 0.0]),
            high=np.array([1.0, 1.0, 1.0]),
            dtype=np.float32
        )
        
        # Observation space: market state + agent state
        n_features = len(self.config['state_features'])
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(n_features,),
            dtype=np.float32
        )
    
    def reset(self, seed: Optional[int] = None, options: Optional[Dict] = None) -> Tuple[np.ndarray, Dict]:
        """
        Reset the environment to initial state.
        
        Args:
            seed: Random seed for reproducibility
            options: Additional options for reset
            
        Returns:
            Initial observation and info dictionary
        """
        super().reset(seed=seed)
        
        # Reset environment state
        self.current_step = 0
        self.capital = self.initial_capital
        self.position = 0.0
        self.portfolio_value = self.initial_capital
        self.trade_history = []
        self.price_history = []
        
        # Reset episode data
        if self.data is not None:
            self.episode_data = self.data.copy()
            self.max_steps = len(self.episode_data)
        else:
            # Generate synthetic data if none provided
            self.episode_data = self._generate_synthetic_data()
            self.max_steps = len(self.episode_data)
        
        # Set episode start time
        self.episode_start_time = datetime.now()
        
        # Get initial observation
        observation = self._get_observation()
        info = self._get_info()
        
        logger.info(f"Environment reset: {self.max_steps} steps, capital: {self.capital}")
        return observation, info
    
    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """
        Execute one step in the environment.
        
        Args:
            action: Action to take [buy_fraction, sell_fraction, hold_probability]
            
        Returns:
            observation, reward, terminated, truncated, info
        """
        if self.current_step >= self.max_steps:
            raise RuntimeError("Environment has not been reset")
        
        # Parse action
        buy_fraction, sell_fraction, hold_probability = action
        
        # Determine actual action based on probabilities
        if hold_probability > 0.5:
            actual_action = 'hold'
        elif buy_fraction > sell_fraction:
            actual_action = 'buy'
            trade_amount = buy_fraction * self.max_position_size
        else:
            actual_action = 'sell'
            trade_amount = sell_fraction * self.max_position_size
        
        # Execute trade
        trade_result = self._execute_trade(actual_action, trade_amount)
        
        # Update state
        self.current_step += 1
        
        # Get new observation
        observation = self._get_observation()
        
        # Calculate reward
        reward = self._calculate_reward(trade_result)
        
        # Check termination conditions
        terminated = self._is_terminated()
        truncated = self._is_truncated()
        
        # Get info
        info = self._get_info()
        info.update(trade_result)
        
        return observation, reward, terminated, truncated, info
    
    def _execute_trade(self, action: str, amount: float) -> Dict:
        """
        Execute a trade in the market.
        
        Args:
            action: 'buy', 'sell', or 'hold'
            amount: Amount to trade in MWh
            
        Returns:
            Dictionary with trade results
        """
        if self.current_step >= len(self.episode_data):
            return {'action': 'hold', 'amount': 0, 'price': 0, 'cost': 0, 'success': False}
        
        # Get current market data
        current_data = self.episode_data.iloc[self.current_step]
        current_price = current_data.get('price', 200.0)  # Default price if not available
        
        # Apply price impact for large trades
        price_impact = self.config.get('price_impact', 0.001)
        if amount > 0:
            price_impact_factor = 1 + (amount / self.max_position_size) * price_impact
            if action == 'buy':
                execution_price = current_price * price_impact_factor
            else:
                execution_price = current_price / price_impact_factor
        else:
            execution_price = current_price
        
        # Calculate trade cost
        trade_cost = amount * execution_price * self.transaction_cost
        
        # Execute trade based on available capital/position
        if action == 'buy':
            total_cost = amount * execution_price + trade_cost
            
            if total_cost <= self.capital and amount > 0:
                # Execute buy
                self.capital -= total_cost
                self.position += amount
                success = True
            else:
                # Insufficient capital
                amount = 0
                success = False
                trade_cost = 0
        
        elif action == 'sell':
            if amount <= self.position and amount > 0:
                # Execute sell
                revenue = amount * execution_price
                self.capital += revenue - trade_cost
                self.position -= amount
                success = True
            else:
                # Insufficient position
                amount = 0
                success = False
                trade_cost = 0
        
        else:  # hold
            amount = 0
            trade_cost = 0
            success = True
        
        # Update portfolio value
        self.portfolio_value = self.capital + self.position * execution_price
        
        # Record trade
        trade_record = {
            'step': self.current_step,
            'action': action,
            'amount': amount,
            'price': execution_price,
            'cost': trade_cost,
            'capital': self.capital,
            'position': self.position,
            'portfolio_value': self.portfolio_value,
            'success': success
        }
        
        self.trade_history.append(trade_record)
        self.price_history.append(execution_price)
        
        return trade_record
    
    def _calculate_reward(self, trade_result: Dict) -> float:
        """
        Calculate reward for the current step.
        
        Args:
            trade_result: Results from the executed trade
            
        Returns:
            Reward value
        """
        return self.reward_calculator.calculate_reward(
            trade_result=trade_result,
            current_state=self._get_state_dict(),
            previous_state=self._get_previous_state()
        )
    
    def _get_observation(self) -> np.ndarray:
        """Get current observation."""
        if self.current_step >= len(self.episode_data):
            # Return zero observation if beyond data
            return np.zeros(self.observation_space.shape[0], dtype=np.float32)
        
        # Get current market data
        current_data = self.episode_data.iloc[self.current_step]
        
        # Build state vector
        state_vector = []
        
        for feature in self.config['state_features']:
            if feature == 'position':
                state_vector.append(self.position / self.max_position_size)  # Normalize
            elif feature == 'capital':
                state_vector.append(self.capital / self.initial_capital)  # Normalize
            elif feature == 'portfolio_value':
                state_vector.append(self.portfolio_value / self.initial_capital)  # Normalize
            elif feature in current_data:
                value = current_data[feature]
                if pd.isna(value):
                    value = 0.0
                state_vector.append(float(value))
            else:
                state_vector.append(0.0)
        
        return np.array(state_vector, dtype=np.float32)
    
    def _get_state_dict(self) -> Dict:
        """Get current state as dictionary."""
        if self.current_step >= len(self.episode_data):
            return {}
        
        current_data = self.episode_data.iloc[self.current_step]
        state_dict = current_data.to_dict()
        state_dict.update({
            'position': self.position,
            'capital': self.capital,
            'portfolio_value': self.portfolio_value,
            'step': self.current_step
        })
        
        return state_dict
    
    def _get_previous_state(self) -> Dict:
        """Get previous state for reward calculation."""
        if self.current_step == 0:
            return {}
        
        if self.current_step - 1 >= len(self.episode_data):
            return {}
        
        previous_data = self.episode_data.iloc[self.current_step - 1]
        return previous_data.to_dict()
    
    def _is_terminated(self) -> bool:
        """Check if episode should be terminated."""
        # Terminate if portfolio value drops below 10% of initial capital
        if self.portfolio_value < self.initial_capital * 0.1:
            return True
        
        # Terminate if we've reached the end of data
        if self.current_step >= self.max_steps:
            return True
        
        return False
    
    def _is_truncated(self) -> bool:
        """Check if episode should be truncated."""
        # Truncate if we've exceeded maximum steps
        return self.current_step >= self.max_steps
    
    def _get_info(self) -> Dict:
        """Get additional information about the current state."""
        return {
            'step': self.current_step,
            'capital': self.capital,
            'position': self.position,
            'portfolio_value': self.portfolio_value,
            'total_trades': len(self.trade_history),
            'episode_duration': datetime.now() - self.episode_start_time if self.episode_start_time else None
        }
    
    def _generate_synthetic_data(self) -> pd.DataFrame:
        """Generate synthetic market data for testing."""
        logger.info("Generating synthetic market data")
        
        # Generate 1000 hours of data
        timestamps = pd.date_range(start='2024-01-01', periods=1000, freq='H')
        
        # Generate price data with realistic patterns
        base_price = 200.0
        price_trend = np.linspace(0, 0.1, 1000)  # Slight upward trend
        price_cycle = 10 * np.sin(2 * np.pi * np.arange(1000) / 24)  # Daily cycle
        price_noise = np.random.normal(0, 5, 1000)  # Random noise
        
        prices = base_price + price_trend + price_cycle + price_noise
        prices = np.maximum(prices, 50.0)  # Minimum price floor
        
        # Generate demand data
        base_demand = 1000.0
        demand_cycle = 200 * np.sin(2 * np.pi * np.arange(1000) / 24)  # Daily demand cycle
        demand_noise = np.random.normal(0, 50, 1000)
        demand = base_demand + demand_cycle + demand_noise
        demand = np.maximum(demand, 100.0)
        
        # Generate supply data (slightly more volatile)
        base_supply = 1100.0
        supply_cycle = 150 * np.sin(2 * np.pi * np.arange(1000) / 24 + np.pi/4)
        supply_noise = np.random.normal(0, 80, 1000)
        supply = base_supply + supply_cycle + supply_noise
        supply = np.maximum(supply, 200.0)
        
        # Create DataFrame
        data = pd.DataFrame({
            'timestamp': timestamps,
            'price': prices,
            'price_change': np.gradient(prices),
            'price_volatility_24h': pd.Series(prices).rolling(24).std().fillna(0),
            'demand': demand,
            'supply': supply,
            'demand_supply_ratio': demand / supply,
            'hour': timestamps.hour,
            'day_of_week': timestamps.dayofweek,
            'is_weekend': (timestamps.dayofweek >= 5).astype(int)
        })
        
        return data
    
    def render(self, mode: str = "human") -> Optional[np.ndarray]:
        """
        Render the environment.
        
        Args:
            mode: Rendering mode ('human' or 'rgb_array')
            
        Returns:
            Rendered image if mode is 'rgb_array', None otherwise
        """
        if mode == "human":
            print(f"Step: {self.current_step}/{self.max_steps}")
            print(f"Capital: {self.capital:.2f} COP")
            print(f"Position: {self.position:.2f} MWh")
            print(f"Portfolio Value: {self.portfolio_value:.2f} COP")
            if self.price_history:
                print(f"Current Price: {self.price_history[-1]:.2f} COP/MWh")
            print(f"Total Trades: {len(self.trade_history)}")
        
        elif mode == "rgb_array":
            # Return a simple visualization as numpy array
            # This would be implemented with matplotlib or similar
            return np.zeros((100, 100, 3), dtype=np.uint8)
        
        return None
    
    def close(self):
        """Close the environment and clean up resources."""
        logger.info("Closing Energy Market Environment")
        # Clean up any resources if needed
        pass
    
    def get_trading_summary(self) -> Dict:
        """Get summary of trading performance."""
        if not self.trade_history:
            return {}
        
        trades_df = pd.DataFrame(self.trade_history)
        
        # Calculate performance metrics
        total_trades = len(trades_df)
        successful_trades = trades_df['success'].sum()
        total_volume = trades_df['amount'].sum()
        total_cost = trades_df['cost'].sum()
        
        # Calculate returns
        initial_value = self.initial_capital
        final_value = self.portfolio_value
        total_return = (final_value - initial_value) / initial_value
        
        # Calculate Sharpe ratio (simplified)
        if len(self.price_history) > 1:
            returns = np.diff(self.price_history) / self.price_history[:-1]
            sharpe_ratio = np.mean(returns) / np.std(returns) if np.std(returns) > 0 else 0
        else:
            sharpe_ratio = 0
        
        return {
            'total_trades': total_trades,
            'successful_trades': successful_trades,
            'success_rate': successful_trades / total_trades if total_trades > 0 else 0,
            'total_volume': total_volume,
            'total_cost': total_cost,
            'initial_capital': initial_value,
            'final_value': final_value,
            'total_return': total_return,
            'sharpe_ratio': sharpe_ratio,
            'max_position': abs(trades_df['position']).max() if not trades_df.empty else 0
        }
