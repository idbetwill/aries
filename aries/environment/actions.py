"""
Action Space Management for Energy Trading Environment

Defines and manages the action space for the reinforcement learning
trading agent, including action encoding and validation.
"""

import numpy as np
from typing import Dict, List, Optional, Tuple, Any, Union
import logging

logger = logging.getLogger(__name__)


class ActionSpace:
    """
    Manages the action space for the energy trading environment.
    
    Handles action encoding, validation, and conversion between
    different action representations.
    """
    
    def __init__(self, 
                 max_position_size: float = 1000.0,
                 action_type: str = 'continuous',
                 config: Dict = None):
        """
        Initialize the action space manager.
        
        Args:
            max_position_size: Maximum position size in MWh
            action_type: Type of action space ('continuous', 'discrete', 'hybrid')
            config: Additional configuration parameters
        """
        self.max_position_size = max_position_size
        self.action_type = action_type
        self.config = config or self._default_config()
        
        # Action space dimensions
        self.n_actions = self._get_action_dimensions()
        
    def _default_config(self) -> Dict:
        """Return default action space configuration."""
        return {
            'discrete_actions': [
                'hold', 'buy_small', 'buy_medium', 'buy_large',
                'sell_small', 'sell_medium', 'sell_large'
            ],
            'continuous_bounds': {
                'buy_fraction': (0.0, 1.0),
                'sell_fraction': (0.0, 1.0),
                'hold_probability': (0.0, 1.0)
            },
            'position_sizes': {
                'small': 0.1,
                'medium': 0.5,
                'large': 1.0
            },
            'action_validation': True,
            'max_trade_frequency': 1.0,  # Maximum trades per hour
            'min_trade_size': 0.01  # Minimum trade size as fraction
        }
    
    def _get_action_dimensions(self) -> int:
        """Get the number of action dimensions."""
        if self.action_type == 'continuous':
            return 3  # [buy_fraction, sell_fraction, hold_probability]
        elif self.action_type == 'discrete':
            return len(self.config['discrete_actions'])
        elif self.action_type == 'hybrid':
            return 4  # [action_type, amount, confidence, timing]
        else:
            raise ValueError(f"Unknown action type: {self.action_type}")
    
    def encode_action(self, action: Union[int, float, np.ndarray, Dict]) -> np.ndarray:
        """
        Encode action into standardized format.
        
        Args:
            action: Action in various formats
            
        Returns:
            Encoded action array
        """
        if self.action_type == 'continuous':
            return self._encode_continuous_action(action)
        elif self.action_type == 'discrete':
            return self._encode_discrete_action(action)
        elif self.action_type == 'hybrid':
            return self._encode_hybrid_action(action)
        else:
            raise ValueError(f"Unknown action type: {self.action_type}")
    
    def _encode_continuous_action(self, action: Union[float, np.ndarray, Dict]) -> np.ndarray:
        """Encode continuous action."""
        if isinstance(action, (int, float)):
            # Single value - assume it's buy/sell amount
            if action > 0:
                return np.array([action, 0.0, 0.0], dtype=np.float32)
            else:
                return np.array([0.0, abs(action), 0.0], dtype=np.float32)
        
        elif isinstance(action, np.ndarray):
            # Array format [buy_fraction, sell_fraction, hold_probability]
            if len(action) >= 3:
                return action[:3].astype(np.float32)
            else:
                # Pad with zeros
                padded = np.zeros(3, dtype=np.float32)
                padded[:len(action)] = action
                return padded
        
        elif isinstance(action, dict):
            # Dictionary format
            buy_fraction = action.get('buy_fraction', 0.0)
            sell_fraction = action.get('sell_fraction', 0.0)
            hold_probability = action.get('hold_probability', 0.0)
            return np.array([buy_fraction, sell_fraction, hold_probability], dtype=np.float32)
        
        else:
            raise ValueError(f"Cannot encode action of type {type(action)}")
    
    def _encode_discrete_action(self, action: Union[int, str]) -> np.ndarray:
        """Encode discrete action."""
        if isinstance(action, int):
            # Integer index
            if 0 <= action < len(self.config['discrete_actions']):
                return np.array([action], dtype=np.int32)
            else:
                raise ValueError(f"Action index {action} out of range")
        
        elif isinstance(action, str):
            # String action name
            if action in self.config['discrete_actions']:
                action_idx = self.config['discrete_actions'].index(action)
                return np.array([action_idx], dtype=np.int32)
            else:
                raise ValueError(f"Unknown action: {action}")
        
        else:
            raise ValueError(f"Cannot encode discrete action of type {type(action)}")
    
    def _encode_hybrid_action(self, action: Dict) -> np.ndarray:
        """Encode hybrid action."""
        action_type = action.get('action_type', 'hold')
        amount = action.get('amount', 0.0)
        confidence = action.get('confidence', 0.5)
        timing = action.get('timing', 0.0)
        
        # Convert action type to numeric
        type_mapping = {'hold': 0, 'buy': 1, 'sell': 2}
        action_type_num = type_mapping.get(action_type, 0)
        
        return np.array([action_type_num, amount, confidence, timing], dtype=np.float32)
    
    def decode_action(self, encoded_action: np.ndarray) -> Dict:
        """
        Decode encoded action back to interpretable format.
        
        Args:
            encoded_action: Encoded action array
            
        Returns:
            Decoded action dictionary
        """
        if self.action_type == 'continuous':
            return self._decode_continuous_action(encoded_action)
        elif self.action_type == 'discrete':
            return self._decode_discrete_action(encoded_action)
        elif self.action_type == 'hybrid':
            return self._decode_hybrid_action(encoded_action)
        else:
            raise ValueError(f"Unknown action type: {self.action_type}")
    
    def _decode_continuous_action(self, encoded_action: np.ndarray) -> Dict:
        """Decode continuous action."""
        buy_fraction, sell_fraction, hold_probability = encoded_action[:3]
        
        # Determine primary action
        if hold_probability > 0.5:
            action_type = 'hold'
            amount = 0.0
        elif buy_fraction > sell_fraction:
            action_type = 'buy'
            amount = buy_fraction * self.max_position_size
        else:
            action_type = 'sell'
            amount = sell_fraction * self.max_position_size
        
        return {
            'action_type': action_type,
            'amount': amount,
            'buy_fraction': buy_fraction,
            'sell_fraction': sell_fraction,
            'hold_probability': hold_probability
        }
    
    def _decode_discrete_action(self, encoded_action: np.ndarray) -> Dict:
        """Decode discrete action."""
        action_idx = int(encoded_action[0])
        action_name = self.config['discrete_actions'][action_idx]
        
        # Convert action name to amount
        if 'buy' in action_name:
            action_type = 'buy'
            size_key = action_name.split('_')[1]
            amount = self.config['position_sizes'][size_key] * self.max_position_size
        elif 'sell' in action_name:
            action_type = 'sell'
            size_key = action_name.split('_')[1]
            amount = self.config['position_sizes'][size_key] * self.max_position_size
        else:  # hold
            action_type = 'hold'
            amount = 0.0
        
        return {
            'action_type': action_type,
            'amount': amount,
            'action_name': action_name
        }
    
    def _decode_hybrid_action(self, encoded_action: np.ndarray) -> Dict:
        """Decode hybrid action."""
        action_type_num, amount, confidence, timing = encoded_action[:4]
        
        # Convert action type number back to string
        type_mapping = {0: 'hold', 1: 'buy', 2: 'sell'}
        action_type = type_mapping.get(int(action_type_num), 'hold')
        
        return {
            'action_type': action_type,
            'amount': amount,
            'confidence': confidence,
            'timing': timing
        }
    
    def validate_action(self, action: Union[np.ndarray, Dict], 
                       current_state: Dict) -> Tuple[bool, str]:
        """
        Validate an action given the current state.
        
        Args:
            action: Action to validate
            current_state: Current market and agent state
            
        Returns:
            Tuple of (is_valid, error_message)
        """
        if not self.config['action_validation']:
            return True, ""
        
        try:
            # Decode action to get interpretable format
            if isinstance(action, np.ndarray):
                decoded_action = self.decode_action(action)
            else:
                decoded_action = action
            
            action_type = decoded_action.get('action_type', 'hold')
            amount = decoded_action.get('amount', 0.0)
            
            # Validate action type
            valid_actions = ['hold', 'buy', 'sell']
            if action_type not in valid_actions:
                return False, f"Invalid action type: {action_type}"
            
            # Validate amount
            if amount < 0:
                return False, "Amount cannot be negative"
            
            if amount > self.max_position_size:
                return False, f"Amount {amount} exceeds maximum position size {self.max_position_size}"
            
            # Validate against current state
            current_position = current_state.get('position', 0)
            current_capital = current_state.get('capital', 0)
            current_price = current_state.get('price', 200)
            
            if action_type == 'sell' and amount > abs(current_position):
                return False, f"Cannot sell {amount} when position is {current_position}"
            
            if action_type == 'buy':
                required_capital = amount * current_price
                if required_capital > current_capital:
                    return False, f"Insufficient capital: need {required_capital}, have {current_capital}"
            
            # Validate trade frequency (simplified)
            # This would need to track trade history in practice
            
            return True, ""
            
        except Exception as e:
            return False, f"Action validation error: {str(e)}"
    
    def get_action_bounds(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get action space bounds.
        
        Returns:
            Tuple of (lower_bounds, upper_bounds)
        """
        if self.action_type == 'continuous':
            lower = np.array([0.0, 0.0, 0.0])
            upper = np.array([1.0, 1.0, 1.0])
        elif self.action_type == 'discrete':
            lower = np.array([0])
            upper = np.array([len(self.config['discrete_actions']) - 1])
        elif self.action_type == 'hybrid':
            lower = np.array([0.0, 0.0, 0.0, 0.0])
            upper = np.array([2.0, 1.0, 1.0, 1.0])
        else:
            raise ValueError(f"Unknown action type: {self.action_type}")
        
        return lower, upper
    
    def sample_action(self) -> np.ndarray:
        """
        Sample a random action from the action space.
        
        Returns:
            Random action array
        """
        if self.action_type == 'continuous':
            return np.random.uniform(0, 1, 3).astype(np.float32)
        elif self.action_type == 'discrete':
            action_idx = np.random.randint(0, len(self.config['discrete_actions']))
            return np.array([action_idx], dtype=np.int32)
        elif self.action_type == 'hybrid':
            action_type = np.random.randint(0, 3)
            amount = np.random.uniform(0, 1)
            confidence = np.random.uniform(0, 1)
            timing = np.random.uniform(0, 1)
            return np.array([action_type, amount, confidence, timing], dtype=np.float32)
        else:
            raise ValueError(f"Unknown action type: {self.action_type}")
    
    def get_action_info(self) -> Dict:
        """Get information about the action space."""
        return {
            'action_type': self.action_type,
            'n_actions': self.n_actions,
            'max_position_size': self.max_position_size,
            'discrete_actions': self.config['discrete_actions'] if self.action_type == 'discrete' else None,
            'bounds': self.get_action_bounds()
        }
