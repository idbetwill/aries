"""
Risk Manager for Trading Agent

Implements comprehensive risk management for the trading agent
including VaR, CVaR, position limits, and dynamic risk adjustment.
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Union, Any
import logging
import warnings
from pathlib import Path
import joblib
from scipy import stats

warnings.filterwarnings('ignore')
logger = logging.getLogger(__name__)


class RiskManager:
    """
    Comprehensive risk management system for the trading agent.
    
    Implements various risk metrics, position limits, and dynamic
    risk adjustment based on market conditions and portfolio state.
    """
    
    def __init__(self, config: Dict = None):
        """
        Initialize the risk manager.
        
        Args:
            config: Configuration dictionary for risk management
        """
        self.config = config or self._default_config()
        self.risk_history = []
        self.position_history = []
        self.portfolio_history = []
        
    def _default_config(self) -> Dict:
        """Return default risk management configuration."""
        return {
            'risk_aversion_lambda': 0.5,
            'max_position_size': 1000.0,
            'max_portfolio_risk': 0.2,
            'var_confidence': 0.05,
            'cvar_confidence': 0.05,
            'max_drawdown': 0.15,
            'stop_loss': 0.1,
            'take_profit': 0.2,
            'position_sizing': {
                'method': 'kelly',  # 'kelly', 'fixed', 'volatility_target'
                'max_fraction': 0.1,
                'volatility_target': 0.15
            },
            'risk_metrics': {
                'var': True,
                'cvar': True,
                'volatility': True,
                'sharpe_ratio': True,
                'max_drawdown': True
            },
            'dynamic_risk': {
                'enabled': True,
                'volatility_threshold': 0.2,
                'correlation_threshold': 0.7,
                'market_regime_detection': True
            }
        }
    
    def adjust_action(self, 
                     action: np.ndarray,
                     observation: np.ndarray,
                     current_state: Dict) -> Tuple[np.ndarray, Dict]:
        """
        Adjust trading action based on risk constraints.
        
        Args:
            action: Original action from RL agent
            observation: Current market observation
            current_state: Current agent state
            
        Returns:
            Tuple of (adjusted_action, risk_info)
        """
        risk_info = {
            'original_action': action.copy(),
            'risk_adjustments': [],
            'risk_metrics': {},
            'constraints_violated': []
        }
        
        # Calculate current risk metrics
        risk_metrics = self.calculate_risk_metrics(
            portfolio_value=current_state.get('portfolio_value', 0),
            position=current_state.get('position', 0),
            price_history=current_state.get('price_history', [])
        )
        risk_info['risk_metrics'] = risk_metrics
        
        # Apply risk adjustments
        adjusted_action = action.copy()
        
        # 1. Position size limits
        adjusted_action = self._apply_position_limits(adjusted_action, current_state, risk_info)
        
        # 2. Portfolio risk limits
        adjusted_action = self._apply_portfolio_risk_limits(adjusted_action, current_state, risk_info)
        
        # 3. Drawdown limits
        adjusted_action = self._apply_drawdown_limits(adjusted_action, current_state, risk_info)
        
        # 4. Volatility-based adjustments
        adjusted_action = self._apply_volatility_adjustments(adjusted_action, observation, risk_info)
        
        # 5. Market regime adjustments
        adjusted_action = self._apply_market_regime_adjustments(adjusted_action, observation, risk_info)
        
        risk_info['adjusted_action'] = adjusted_action
        risk_info['adjustment_magnitude'] = np.linalg.norm(adjusted_action - action)
        
        return adjusted_action, risk_info
    
    def _apply_position_limits(self, action: np.ndarray, current_state: Dict, risk_info: Dict) -> np.ndarray:
        """Apply position size limits."""
        max_position = self.config['max_position_size']
        current_position = current_state.get('position', 0)
        
        # Parse action (assuming [buy_fraction, sell_fraction, hold_probability])
        buy_fraction, sell_fraction, hold_probability = action
        
        # Calculate proposed position change
        max_trade_size = max_position * 0.1  # Maximum 10% of max position per trade
        proposed_buy = buy_fraction * max_trade_size
        proposed_sell = sell_fraction * max_trade_size
        proposed_position = current_position + proposed_buy - proposed_sell
        
        # Check position limits
        if abs(proposed_position) > max_position:
            # Scale down the action
            scale_factor = max_position / abs(proposed_position) if proposed_position != 0 else 1.0
            action[0] *= scale_factor  # buy_fraction
            action[1] *= scale_factor  # sell_fraction
            
            risk_info['risk_adjustments'].append('position_limit')
            risk_info['constraints_violated'].append('max_position')
        
        return action
    
    def _apply_portfolio_risk_limits(self, action: np.ndarray, current_state: Dict, risk_info: Dict) -> np.ndarray:
        """Apply portfolio risk limits."""
        max_risk = self.config['max_portfolio_risk']
        portfolio_value = current_state.get('portfolio_value', 0)
        
        if portfolio_value <= 0:
            return action
        
        # Calculate current portfolio risk
        risk_metrics = risk_info['risk_metrics']
        current_risk = risk_metrics.get('portfolio_risk', 0)
        
        if current_risk > max_risk:
            # Reduce position size based on risk
            risk_reduction_factor = max_risk / current_risk
            action[0] *= risk_reduction_factor  # buy_fraction
            action[1] *= risk_reduction_factor  # sell_fraction
            
            risk_info['risk_adjustments'].append('portfolio_risk_limit')
            risk_info['constraints_violated'].append('max_portfolio_risk')
        
        return action
    
    def _apply_drawdown_limits(self, action: np.ndarray, current_state: Dict, risk_info: Dict) -> np.ndarray:
        """Apply drawdown limits."""
        max_drawdown = self.config['max_drawdown']
        portfolio_value = current_state.get('portfolio_value', 0)
        initial_capital = current_state.get('initial_capital', portfolio_value)
        
        if initial_capital <= 0:
            return action
        
        # Calculate current drawdown
        current_drawdown = (initial_capital - portfolio_value) / initial_capital
        
        if current_drawdown > max_drawdown:
            # Stop trading or reduce position
            action[0] *= 0.1  # Significantly reduce buy
            action[1] *= 0.1  # Significantly reduce sell
            action[2] = 0.9   # Increase hold probability
            
            risk_info['risk_adjustments'].append('drawdown_limit')
            risk_info['constraints_violated'].append('max_drawdown')
        
        return action
    
    def _apply_volatility_adjustments(self, action: np.ndarray, observation: np.ndarray, risk_info: Dict) -> np.ndarray:
        """Apply volatility-based adjustments."""
        # Extract volatility from observation (assuming it's in the observation)
        if len(observation) > 2:
            volatility = observation[2]  # Assuming volatility is the 3rd feature
        else:
            volatility = 0.1  # Default volatility
        
        volatility_threshold = self.config['dynamic_risk']['volatility_threshold']
        
        if volatility > volatility_threshold:
            # Reduce position size in high volatility
            volatility_factor = volatility_threshold / volatility
            action[0] *= volatility_factor  # buy_fraction
            action[1] *= volatility_factor  # sell_fraction
            
            risk_info['risk_adjustments'].append('volatility_adjustment')
        
        return action
    
    def _apply_market_regime_adjustments(self, action: np.ndarray, observation: np.ndarray, risk_info: Dict) -> np.ndarray:
        """Apply market regime-based adjustments."""
        if not self.config['dynamic_risk']['market_regime_detection']:
            return action
        
        # Simple market regime detection based on observation
        # This is a placeholder - in practice, you'd use more sophisticated methods
        market_stress = np.mean(observation[:3])  # Average of first 3 features
        
        if market_stress > 0.5:  # High stress market
            # Reduce position size
            action[0] *= 0.5  # buy_fraction
            action[1] *= 0.5  # sell_fraction
            action[2] = min(action[2] + 0.2, 1.0)  # Increase hold probability
            
            risk_info['risk_adjustments'].append('market_regime_adjustment')
        
        return action
    
    def calculate_risk_metrics(self, 
                              portfolio_value: float,
                              position: float,
                              price_history: List[float]) -> Dict:
        """Calculate comprehensive risk metrics."""
        metrics = {}
        
        # Basic metrics
        metrics['position_size'] = abs(position)
        metrics['position_ratio'] = abs(position) / self.config['max_position_size'] if self.config['max_position_size'] > 0 else 0
        
        # Portfolio risk
        if portfolio_value > 0:
            metrics['portfolio_risk'] = abs(position) / portfolio_value
        else:
            metrics['portfolio_risk'] = 0
        
        # Value at Risk (VaR)
        if len(price_history) > 10:
            returns = np.diff(price_history) / price_history[:-1]
            var_confidence = self.config['var_confidence']
            metrics['var'] = np.percentile(returns, var_confidence * 100)
        else:
            metrics['var'] = 0
        
        # Conditional Value at Risk (CVaR)
        if len(price_history) > 10:
            returns = np.diff(price_history) / price_history[:-1]
            var = metrics['var']
            tail_returns = returns[returns <= var]
            metrics['cvar'] = np.mean(tail_returns) if len(tail_returns) > 0 else var
        else:
            metrics['cvar'] = 0
        
        # Volatility
        if len(price_history) > 10:
            returns = np.diff(price_history) / price_history[:-1]
            metrics['volatility'] = np.std(returns)
        else:
            metrics['volatility'] = 0
        
        # Sharpe ratio
        if len(price_history) > 10:
            returns = np.diff(price_history) / price_history[:-1]
            if metrics['volatility'] > 0:
                metrics['sharpe_ratio'] = np.mean(returns) / metrics['volatility']
            else:
                metrics['sharpe_ratio'] = 0
        else:
            metrics['sharpe_ratio'] = 0
        
        # Maximum drawdown
        if len(price_history) > 1:
            running_max = np.maximum.accumulate(price_history)
            drawdown = (price_history - running_max) / running_max
            metrics['max_drawdown'] = np.min(drawdown)
        else:
            metrics['max_drawdown'] = 0
        
        return metrics
    
    def calculate_comprehensive_risk_metrics(self, 
                                           portfolio_values: List[float],
                                           returns: List[float],
                                           positions: List[float]) -> Dict:
        """Calculate comprehensive risk metrics from historical data."""
        if len(portfolio_values) < 2:
            return {}
        
        portfolio_array = np.array(portfolio_values)
        returns_array = np.array(returns) if returns else np.diff(portfolio_values)
        positions_array = np.array(positions)
        
        metrics = {}
        
        # Portfolio metrics
        total_return = (portfolio_array[-1] - portfolio_array[0]) / portfolio_array[0]
        metrics['total_return'] = total_return
        
        # Risk metrics
        if len(returns_array) > 1:
            metrics['volatility'] = np.std(returns_array)
            metrics['sharpe_ratio'] = np.mean(returns_array) / np.std(returns_array) if np.std(returns_array) > 0 else 0
        
        # Drawdown metrics
        running_max = np.maximum.accumulate(portfolio_array)
        drawdown = (portfolio_array - running_max) / running_max
        metrics['max_drawdown'] = np.min(drawdown)
        metrics['current_drawdown'] = drawdown[-1]
        
        # VaR and CVaR
        if len(returns_array) > 10:
            var_95 = np.percentile(returns_array, 5)
            cvar_95 = np.mean(returns_array[returns_array <= var_95])
            metrics['var_95'] = var_95
            metrics['cvar_95'] = cvar_95
        
        # Position metrics
        metrics['max_position'] = np.max(np.abs(positions_array))
        metrics['avg_position'] = np.mean(np.abs(positions_array))
        metrics['position_volatility'] = np.std(positions_array)
        
        # Risk-adjusted metrics
        if metrics.get('volatility', 0) > 0:
            metrics['calmar_ratio'] = total_return / abs(metrics['max_drawdown']) if metrics['max_drawdown'] != 0 else 0
            metrics['sortino_ratio'] = np.mean(returns_array) / np.std(returns_array[returns_array < 0]) if np.std(returns_array[returns_array < 0]) > 0 else 0
        
        return metrics
    
    def update_config(self, new_config: Dict):
        """Update risk management configuration."""
        self.config.update(new_config)
        logger.info("Risk management configuration updated")
    
    def get_info(self) -> Dict:
        """Get risk manager information."""
        return {
            'config': self.config,
            'risk_history_length': len(self.risk_history),
            'position_history_length': len(self.position_history),
            'portfolio_history_length': len(self.portfolio_history)
        }
    
    def save(self, path: str):
        """Save risk manager state."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        risk_state = {
            'config': self.config,
            'risk_history': self.risk_history,
            'position_history': self.position_history,
            'portfolio_history': self.portfolio_history
        }
        
        joblib.dump(risk_state, path)
        logger.info(f"Risk manager saved to {path}")
    
    def load(self, path: str):
        """Load risk manager state."""
        path = Path(path)
        
        if not path.exists():
            raise FileNotFoundError(f"Risk manager file not found: {path}")
        
        risk_state = joblib.load(path)
        
        self.config = risk_state['config']
        self.risk_history = risk_state.get('risk_history', [])
        self.position_history = risk_state.get('position_history', [])
        self.portfolio_history = risk_state.get('portfolio_history', [])
        
        logger.info(f"Risk manager loaded from {path}")
    
    def add_risk_record(self, risk_metrics: Dict, position: float, portfolio_value: float):
        """Add a risk record to history."""
        record = {
            'timestamp': datetime.now(),
            'risk_metrics': risk_metrics,
            'position': position,
            'portfolio_value': portfolio_value
        }
        
        self.risk_history.append(record)
        self.position_history.append(position)
        self.portfolio_history.append(portfolio_value)
        
        # Keep only recent history (last 1000 records)
        if len(self.risk_history) > 1000:
            self.risk_history = self.risk_history[-1000:]
            self.position_history = self.position_history[-1000:]
            self.portfolio_history = self.portfolio_history[-1000:]
