"""
Risk-Averse Trading Agent

Main RL agent implementation that combines reinforcement learning
with risk management for energy market trading.
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Union, Any
import logging
import warnings
from pathlib import Path
import joblib

from .ppo_agent import PPOAgent
from .sac_agent import SACAgent
from .risk_manager import RiskManager
from .training import TrainingManager

warnings.filterwarnings('ignore')
logger = logging.getLogger(__name__)


class RiskAverseTradingAgent:
    """
    Risk-averse trading agent that combines RL with risk management.
    
    Implements sophisticated risk-averse decision making for energy market
    trading using reinforcement learning algorithms with risk constraints.
    """
    
    def __init__(self, 
                 environment,
                 forecaster,
                 config: Dict = None,
                 algorithm: str = 'PPO'):
        """
        Initialize the risk-averse trading agent.
        
        Args:
            environment: Trading environment (Gymnasium)
            forecaster: Probabilistic forecaster
            config: Configuration dictionary
            algorithm: RL algorithm to use ('PPO' or 'SAC')
        """
        self.environment = environment
        self.forecaster = forecaster
        self.config = config or self._default_config()
        self.algorithm = algorithm
        
        # Initialize components
        self.risk_manager = RiskManager(config=self.config.get('risk', {}))
        self.training_manager = TrainingManager(config=self.config.get('training', {}))
        
        # Initialize RL agent
        self._initialize_rl_agent()
        
        # Agent state
        self.is_trained = False
        self.training_history = []
        self.performance_metrics = {}
        
    def _default_config(self) -> Dict:
        """Return default agent configuration."""
        return {
            'algorithm': 'PPO',
            'risk': {
                'risk_aversion_lambda': 0.5,
                'max_position_size': 1000.0,
                'var_confidence': 0.05,
                'cvar_confidence': 0.05,
                'max_drawdown': 0.2,
                'stop_loss': 0.1
            },
            'training': {
                'total_timesteps': 100000,
                'learning_rate': 3e-4,
                'batch_size': 64,
                'n_epochs': 10,
                'gamma': 0.99,
                'gae_lambda': 0.95,
                'clip_range': 0.2,
                'ent_coef': 0.01,
                'vf_coef': 0.5,
                'max_grad_norm': 0.5
            },
            'forecasting': {
                'forecast_horizon': 24,
                'update_frequency': 1,  # hours
                'uncertainty_threshold': 0.1
            },
            'trading': {
                'min_trade_size': 0.01,
                'max_trade_size': 1.0,
                'transaction_cost': 0.001,
                'slippage': 0.0005
            }
        }
    
    def _initialize_rl_agent(self):
        """Initialize the RL agent based on algorithm choice."""
        if self.algorithm.upper() == 'PPO':
            self.rl_agent = PPOAgent(
                env=self.environment,
                config=self.config['training']
            )
        elif self.algorithm.upper() == 'SAC':
            self.rl_agent = SACAgent(
                env=self.environment,
                config=self.config['training']
            )
        else:
            raise ValueError(f"Unknown algorithm: {self.algorithm}")
        
        logger.info(f"Initialized {self.algorithm} agent")
    
    def train(self, 
              data: pd.DataFrame = None,
              episodes: int = None,
              callbacks: List = None) -> Dict:
        """
        Train the risk-averse trading agent.
        
        Args:
            data: Training data (if None, uses environment data)
            episodes: Number of training episodes
            callbacks: Training callbacks
            
        Returns:
            Training results dictionary
        """
        logger.info("Starting risk-averse agent training")
        
        # Prepare training data
        if data is not None:
            self.environment.data = data
        
        # Set training parameters
        total_timesteps = episodes * self.environment.max_steps if episodes else self.config['training']['total_timesteps']
        
        # Train the RL agent
        training_results = self.rl_agent.train(
            total_timesteps=total_timesteps,
            callbacks=callbacks
        )
        
        # Store training history
        self.training_history.append({
            'timestamp': datetime.now(),
            'episodes': episodes,
            'total_timesteps': total_timesteps,
            'results': training_results
        })
        
        self.is_trained = True
        logger.info("Risk-averse agent training completed")
        
        return training_results
    
    def predict(self, observation: np.ndarray, deterministic: bool = True) -> Tuple[np.ndarray, Dict]:
        """
        Make a trading decision based on current observation.
        
        Args:
            observation: Current market observation
            deterministic: Whether to use deterministic policy
            
        Returns:
            Tuple of (action, info)
        """
        if not self.is_trained:
            raise RuntimeError("Agent must be trained before making predictions")
        
        # Get base action from RL agent
        action, rl_info = self.rl_agent.predict(observation, deterministic=deterministic)
        
        # Apply risk management
        risk_adjusted_action, risk_info = self.risk_manager.adjust_action(
            action=action,
            observation=observation,
            current_state=self._get_current_state()
        )
        
        # Combine information
        info = {
            'rl_action': action,
            'risk_adjusted_action': risk_adjusted_action,
            'rl_info': rl_info,
            'risk_info': risk_info
        }
        
        return risk_adjusted_action, info
    
    def _get_current_state(self) -> Dict:
        """Get current agent state for risk management."""
        # This would typically come from the environment
        # For now, return a placeholder
        return {
            'position': 0.0,
            'capital': 100000.0,
            'portfolio_value': 100000.0,
            'risk_metrics': {}
        }
    
    def backtest(self, 
                 test_data: pd.DataFrame,
                 initial_capital: float = 100000.0,
                 verbose: bool = True) -> Dict:
        """
        Run backtesting on historical data.
        
        Args:
            test_data: Historical data for backtesting
            initial_capital: Initial capital for backtesting
            verbose: Whether to print progress
            
        Returns:
            Backtesting results dictionary
        """
        if not self.is_trained:
            raise RuntimeError("Agent must be trained before backtesting")
        
        logger.info("Starting backtesting")
        
        # Set up backtesting environment
        self.environment.data = test_data
        self.environment.capital = initial_capital
        self.environment.portfolio_value = initial_capital
        
        # Run backtesting
        obs, _ = self.environment.reset()
        done = False
        step = 0
        
        backtest_results = {
            'portfolio_values': [],
            'positions': [],
            'trades': [],
            'rewards': [],
            'risk_metrics': []
        }
        
        while not done:
            # Make prediction
            action, info = self.predict(obs, deterministic=True)
            
            # Execute action
            obs, reward, terminated, truncated, step_info = self.environment.step(action)
            done = terminated or truncated
            
            # Record results
            backtest_results['portfolio_values'].append(self.environment.portfolio_value)
            backtest_results['positions'].append(self.environment.position)
            backtest_results['trades'].append(step_info)
            backtest_results['rewards'].append(reward)
            
            # Calculate risk metrics
            risk_metrics = self.risk_manager.calculate_risk_metrics(
                portfolio_value=self.environment.portfolio_value,
                position=self.environment.position,
                price_history=backtest_results['portfolio_values']
            )
            backtest_results['risk_metrics'].append(risk_metrics)
            
            step += 1
            
            if verbose and step % 100 == 0:
                logger.info(f"Backtesting step {step}: Portfolio = {self.environment.portfolio_value:.2f}")
        
        # Calculate performance metrics
        performance = self._calculate_performance_metrics(backtest_results)
        backtest_results['performance'] = performance
        
        logger.info("Backtesting completed")
        return backtest_results
    
    def _calculate_performance_metrics(self, backtest_results: Dict) -> Dict:
        """Calculate performance metrics from backtesting results."""
        portfolio_values = np.array(backtest_results['portfolio_values'])
        rewards = np.array(backtest_results['rewards'])
        
        if len(portfolio_values) == 0:
            return {}
        
        # Basic metrics
        initial_value = portfolio_values[0]
        final_value = portfolio_values[-1]
        total_return = (final_value - initial_value) / initial_value
        
        # Risk metrics
        returns = np.diff(portfolio_values) / portfolio_values[:-1]
        volatility = np.std(returns) if len(returns) > 1 else 0
        
        # Sharpe ratio
        sharpe_ratio = np.mean(returns) / volatility if volatility > 0 else 0
        
        # Maximum drawdown
        running_max = np.maximum.accumulate(portfolio_values)
        drawdown = (portfolio_values - running_max) / running_max
        max_drawdown = np.min(drawdown)
        
        # VaR and CVaR
        var_95 = np.percentile(returns, 5) if len(returns) > 0 else 0
        cvar_95 = np.mean(returns[returns <= var_95]) if len(returns) > 0 else 0
        
        # Trading metrics
        trades = backtest_results['trades']
        successful_trades = sum(1 for trade in trades if trade.get('success', False))
        total_trades = len(trades)
        success_rate = successful_trades / total_trades if total_trades > 0 else 0
        
        return {
            'total_return': total_return,
            'volatility': volatility,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'var_95': var_95,
            'cvar_95': cvar_95,
            'success_rate': success_rate,
            'total_trades': total_trades,
            'final_portfolio_value': final_value
        }
    
    def get_forecast(self, current_data: pd.DataFrame) -> Dict:
        """
        Get probabilistic forecast for current market conditions.
        
        Args:
            current_data: Current market data
            
        Returns:
            Forecast dictionary
        """
        if not self.forecaster.is_trained:
            logger.warning("Forecaster not trained, returning empty forecast")
            return {}
        
        try:
            forecast = self.forecaster.predict(current_data)
            return forecast
        except Exception as e:
            logger.error(f"Error getting forecast: {e}")
            return {}
    
    def update_risk_parameters(self, risk_config: Dict):
        """
        Update risk management parameters.
        
        Args:
            risk_config: New risk configuration
        """
        self.config['risk'].update(risk_config)
        self.risk_manager.update_config(risk_config)
        logger.info("Risk parameters updated")
    
    def get_agent_info(self) -> Dict:
        """Get information about the agent."""
        return {
            'is_trained': self.is_trained,
            'algorithm': self.algorithm,
            'config': self.config,
            'training_history': len(self.training_history),
            'risk_manager_info': self.risk_manager.get_info(),
            'rl_agent_info': self.rl_agent.get_info() if hasattr(self.rl_agent, 'get_info') else {}
        }
    
    def save_agent(self, path: str):
        """Save the trained agent."""
        if not self.is_trained:
            raise RuntimeError("Agent must be trained before saving")
        
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        
        # Save RL agent
        rl_agent_path = path / "rl_agent.pkl"
        self.rl_agent.save(str(rl_agent_path))
        
        # Save risk manager
        risk_manager_path = path / "risk_manager.pkl"
        self.risk_manager.save(str(risk_manager_path))
        
        # Save agent state
        agent_state = {
            'config': self.config,
            'algorithm': self.algorithm,
            'is_trained': self.is_trained,
            'training_history': self.training_history,
            'performance_metrics': self.performance_metrics
        }
        
        agent_path = path / "agent_state.pkl"
        joblib.dump(agent_state, agent_path)
        
        logger.info(f"Agent saved to {path}")
    
    def load_agent(self, path: str):
        """Load a trained agent."""
        path = Path(path)
        
        if not path.exists():
            raise FileNotFoundError(f"Agent path not found: {path}")
        
        # Load agent state
        agent_path = path / "agent_state.pkl"
        if agent_path.exists():
            agent_state = joblib.load(agent_path)
            self.config = agent_state['config']
            self.algorithm = agent_state['algorithm']
            self.training_history = agent_state.get('training_history', [])
            self.performance_metrics = agent_state.get('performance_metrics', {})
        
        # Load RL agent
        rl_agent_path = path / "rl_agent.pkl"
        if rl_agent_path.exists():
            self.rl_agent.load(str(rl_agent_path))
        
        # Load risk manager
        risk_manager_path = path / "risk_manager.pkl"
        if risk_manager_path.exists():
            self.risk_manager.load(str(risk_manager_path))
        
        self.is_trained = True
        logger.info(f"Agent loaded from {path}")
    
    def evaluate_performance(self, test_data: pd.DataFrame) -> Dict:
        """
        Evaluate agent performance on test data.
        
        Args:
            test_data: Test data for evaluation
            
        Returns:
            Performance evaluation results
        """
        # Run backtesting
        backtest_results = self.backtest(test_data, verbose=False)
        
        # Calculate additional metrics
        performance = backtest_results['performance']
        
        # Add risk-adjusted metrics
        risk_metrics = self.risk_manager.calculate_comprehensive_risk_metrics(
            portfolio_values=backtest_results['portfolio_values'],
            returns=np.diff(backtest_results['portfolio_values']),
            positions=backtest_results['positions']
        )
        
        performance.update(risk_metrics)
        
        # Store performance metrics
        self.performance_metrics = performance
        
        return performance
