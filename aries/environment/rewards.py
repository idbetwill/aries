"""
Reward Calculation for Risk-Averse Trading Agent

Implements sophisticated reward functions that incorporate risk aversion,
transaction costs, and market dynamics for the RL trading agent.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any
import logging
from scipy import stats
import warnings

warnings.filterwarnings('ignore')
logger = logging.getLogger(__name__)


class RewardCalculator:
    """
    Calculates rewards for the risk-averse trading agent.
    
    Implements multiple reward components including profit/loss,
    risk penalties, transaction costs, and market regime adjustments.
    """
    
    def __init__(self, 
                 risk_aversion_lambda: float = 0.5,
                 transaction_cost: float = 0.001,
                 config: Dict = None):
        """
        Initialize the reward calculator.
        
        Args:
            risk_aversion_lambda: Risk aversion coefficient
            transaction_cost: Transaction cost as percentage
            config: Additional configuration parameters
        """
        self.risk_aversion_lambda = risk_aversion_lambda
        self.transaction_cost = transaction_cost
        self.config = config or self._default_config()
        
        # Reward history for tracking
        self.reward_history = []
        self.risk_history = []
        
    def _default_config(self) -> Dict:
        """Return default reward configuration."""
        return {
            'reward_components': {
                'profit_loss': 1.0,
                'risk_penalty': 1.0,
                'transaction_cost': 1.0,
                'market_regime': 0.5,
                'position_penalty': 0.3,
                'volatility_penalty': 0.2
            },
            'risk_metrics': ['var', 'cvar', 'volatility'],
            'var_confidence': 0.05,  # 5% VaR
            'cvar_confidence': 0.05,  # 5% CVaR
            'lookback_period': 24,  # hours
            'volatility_window': 24,  # hours
            'max_position_penalty': 0.1,
            'volatility_threshold': 0.1
        }
    
    def calculate_reward(self, 
                        trade_result: Dict,
                        current_state: Dict,
                        previous_state: Dict = None) -> float:
        """
        Calculate the reward for a trading action.
        
        Args:
            trade_result: Results from the executed trade
            current_state: Current market and agent state
            previous_state: Previous state for comparison
            
        Returns:
            Calculated reward value
        """
        # Initialize reward components
        reward_components = {}
        
        # 1. Profit/Loss component
        profit_loss = self._calculate_profit_loss_reward(trade_result, current_state, previous_state)
        reward_components['profit_loss'] = profit_loss
        
        # 2. Risk penalty component
        risk_penalty = self._calculate_risk_penalty(current_state, trade_result)
        reward_components['risk_penalty'] = risk_penalty
        
        # 3. Transaction cost component
        transaction_cost_penalty = self._calculate_transaction_cost_penalty(trade_result)
        reward_components['transaction_cost'] = transaction_cost_penalty
        
        # 4. Market regime component
        market_regime_reward = self._calculate_market_regime_reward(current_state)
        reward_components['market_regime'] = market_regime_reward
        
        # 5. Position penalty component
        position_penalty = self._calculate_position_penalty(current_state)
        reward_components['position_penalty'] = position_penalty
        
        # 6. Volatility penalty component
        volatility_penalty = self._calculate_volatility_penalty(current_state)
        reward_components['volatility_penalty'] = volatility_penalty
        
        # Combine reward components
        total_reward = self._combine_reward_components(reward_components)
        
        # Store for analysis
        self.reward_history.append({
            'total_reward': total_reward,
            'components': reward_components,
            'timestamp': current_state.get('timestamp', None)
        })
        
        return total_reward
    
    def _calculate_profit_loss_reward(self, 
                                     trade_result: Dict,
                                     current_state: Dict,
                                     previous_state: Dict) -> float:
        """Calculate profit/loss based reward component."""
        if not trade_result.get('success', False):
            return -0.1  # Small penalty for failed trades
        
        # Calculate immediate profit/loss from the trade
        trade_amount = trade_result.get('amount', 0)
        trade_price = trade_result.get('price', 0)
        trade_cost = trade_result.get('cost', 0)
        
        if trade_amount == 0:  # Hold action
            return 0.0
        
        # Calculate profit/loss
        if trade_result.get('action') == 'buy':
            # For buying, profit comes from future price appreciation
            # This is a simplified model - in reality, we'd need future prices
            profit = 0.0  # Placeholder for future price appreciation
        else:  # sell
            # For selling, profit comes from the price difference
            # This is also simplified
            profit = 0.0  # Placeholder for price difference
        
        # Normalize by position size
        if trade_amount > 0:
            normalized_profit = profit / trade_amount
        else:
            normalized_profit = 0.0
        
        return normalized_profit
    
    def _calculate_risk_penalty(self, current_state: Dict, trade_result: Dict) -> float:
        """Calculate risk-based penalty using VaR and CVaR."""
        # Get current portfolio metrics
        portfolio_value = current_state.get('portfolio_value', 0)
        position = current_state.get('position', 0)
        
        if portfolio_value == 0:
            return 0.0
        
        # Calculate portfolio risk metrics
        risk_metrics = self._calculate_portfolio_risk(current_state)
        
        # VaR penalty
        var_penalty = risk_metrics.get('var', 0) * self.risk_aversion_lambda
        
        # CVaR penalty (more severe)
        cvar_penalty = risk_metrics.get('cvar', 0) * self.risk_aversion_lambda * 1.5
        
        # Volatility penalty
        volatility_penalty = risk_metrics.get('volatility', 0) * self.risk_aversion_lambda * 0.5
        
        total_risk_penalty = -(var_penalty + cvar_penalty + volatility_penalty)
        
        return total_risk_penalty
    
    def _calculate_portfolio_risk(self, current_state: Dict) -> Dict:
        """Calculate portfolio risk metrics."""
        # This is a simplified risk calculation
        # In practice, you'd use historical returns and more sophisticated models
        
        portfolio_value = current_state.get('portfolio_value', 0)
        position = current_state.get('position', 0)
        price = current_state.get('price', 200)
        
        if portfolio_value == 0:
            return {'var': 0, 'cvar': 0, 'volatility': 0}
        
        # Simplified risk calculation based on position size
        position_ratio = abs(position) / (portfolio_value / price) if price > 0 else 0
        
        # VaR (simplified)
        var = position_ratio * 0.05  # 5% VaR
        
        # CVaR (simplified)
        cvar = position_ratio * 0.08  # 8% CVaR
        
        # Volatility (simplified)
        volatility = position_ratio * 0.1  # 10% volatility
        
        return {
            'var': var,
            'cvar': cvar,
            'volatility': volatility
        }
    
    def _calculate_transaction_cost_penalty(self, trade_result: Dict) -> float:
        """Calculate penalty for transaction costs."""
        trade_cost = trade_result.get('cost', 0)
        trade_amount = trade_result.get('amount', 0)
        
        if trade_amount == 0:
            return 0.0
        
        # Normalize cost by trade amount
        normalized_cost = trade_cost / trade_amount if trade_amount > 0 else 0
        
        # Penalty proportional to transaction cost
        cost_penalty = -normalized_cost * 0.1
        
        return cost_penalty
    
    def _calculate_market_regime_reward(self, current_state: Dict) -> float:
        """Calculate reward based on market regime."""
        # Get market indicators
        price_volatility = current_state.get('price_volatility_24h', 0)
        demand_supply_ratio = current_state.get('demand_supply_ratio', 1)
        hour = current_state.get('hour', 12)
        
        # Regime-based rewards
        if price_volatility > 0.1:  # High volatility
            return -0.2  # Penalty for trading in high volatility
        elif demand_supply_ratio > 1.2:  # High demand
            return 0.1  # Reward for trading in high demand
        elif demand_supply_ratio < 0.8:  # Low demand
            return -0.1  # Penalty for trading in low demand
        elif 8 <= hour <= 18:  # Business hours
            return 0.05  # Small reward for trading during business hours
        else:
            return 0.0  # Neutral for normal conditions
    
    def _calculate_position_penalty(self, current_state: Dict) -> float:
        """Calculate penalty for large positions."""
        position = current_state.get('position', 0)
        max_position = current_state.get('max_position_size', 1000)
        
        if max_position == 0:
            return 0.0
        
        # Calculate position ratio
        position_ratio = abs(position) / max_position
        
        # Penalty increases quadratically with position size
        position_penalty = -(position_ratio ** 2) * self.config['max_position_penalty']
        
        return position_penalty
    
    def _calculate_volatility_penalty(self, current_state: Dict) -> float:
        """Calculate penalty for high volatility periods."""
        price_volatility = current_state.get('price_volatility_24h', 0)
        volatility_threshold = self.config['volatility_threshold']
        
        if price_volatility > volatility_threshold:
            # Penalty proportional to excess volatility
            excess_volatility = price_volatility - volatility_threshold
            volatility_penalty = -excess_volatility * 0.5
            return volatility_penalty
        
        return 0.0
    
    def _combine_reward_components(self, components: Dict) -> float:
        """Combine different reward components."""
        total_reward = 0.0
        
        for component, value in components.items():
            weight = self.config['reward_components'].get(component, 1.0)
            total_reward += weight * value
        
        return total_reward
    
    def calculate_sharpe_ratio(self, returns: List[float]) -> float:
        """Calculate Sharpe ratio from returns."""
        if len(returns) < 2:
            return 0.0
        
        returns_array = np.array(returns)
        mean_return = np.mean(returns_array)
        std_return = np.std(returns_array)
        
        if std_return == 0:
            return 0.0
        
        # Assume risk-free rate of 0 for simplicity
        sharpe_ratio = mean_return / std_return
        
        return sharpe_ratio
    
    def calculate_max_drawdown(self, portfolio_values: List[float]) -> float:
        """Calculate maximum drawdown from portfolio values."""
        if len(portfolio_values) < 2:
            return 0.0
        
        values = np.array(portfolio_values)
        peak = np.maximum.accumulate(values)
        drawdown = (values - peak) / peak
        
        return np.min(drawdown)
    
    def calculate_var(self, returns: List[float], confidence: float = 0.05) -> float:
        """Calculate Value at Risk (VaR)."""
        if len(returns) < 10:
            return 0.0
        
        returns_array = np.array(returns)
        var = np.percentile(returns_array, confidence * 100)
        
        return var
    
    def calculate_cvar(self, returns: List[float], confidence: float = 0.05) -> float:
        """Calculate Conditional Value at Risk (CVaR)."""
        if len(returns) < 10:
            return 0.0
        
        returns_array = np.array(returns)
        var = self.calculate_var(returns, confidence)
        
        # CVaR is the mean of returns below VaR
        tail_returns = returns_array[returns_array <= var]
        
        if len(tail_returns) == 0:
            return var
        
        cvar = np.mean(tail_returns)
        
        return cvar
    
    def get_reward_summary(self) -> Dict:
        """Get summary of reward calculations."""
        if not self.reward_history:
            return {}
        
        # Calculate statistics
        total_rewards = [r['total_reward'] for r in self.reward_history]
        
        summary = {
            'total_rewards': len(total_rewards),
            'mean_reward': np.mean(total_rewards),
            'std_reward': np.std(total_rewards),
            'min_reward': np.min(total_rewards),
            'max_reward': np.max(total_rewards),
            'cumulative_reward': np.sum(total_rewards)
        }
        
        # Component analysis
        components = {}
        for component in self.config['reward_components'].keys():
            component_values = [r['components'].get(component, 0) for r in self.reward_history]
            components[component] = {
                'mean': np.mean(component_values),
                'std': np.std(component_values),
                'total': np.sum(component_values)
            }
        
        summary['components'] = components
        
        return summary
    
    def reset_history(self):
        """Reset reward history."""
        self.reward_history = []
        self.risk_history = []
