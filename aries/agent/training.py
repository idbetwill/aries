"""
Training Manager for RL Agent

Handles training orchestration, hyperparameter optimization,
and training monitoring for the risk-averse trading agent.
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Union, Any
import logging
import warnings
from pathlib import Path
import joblib
import json

warnings.filterwarnings('ignore')
logger = logging.getLogger(__name__)


class TrainingManager:
    """
    Training manager for the risk-averse trading agent.
    
    Handles training orchestration, hyperparameter optimization,
    and comprehensive training monitoring.
    """
    
    def __init__(self, config: Dict = None):
        """
        Initialize the training manager.
        
        Args:
            config: Configuration dictionary for training
        """
        self.config = config or self._default_config()
        self.training_history = []
        self.best_models = {}
        self.hyperparameter_results = {}
        
    def _default_config(self) -> Dict:
        """Return default training configuration."""
        return {
            'training': {
                'total_timesteps': 100000,
                'eval_freq': 10000,
                'save_freq': 50000,
                'log_interval': 1000,
                'verbose': 1
            },
            'hyperparameter_optimization': {
                'enabled': False,
                'n_trials': 20,
                'optimization_metric': 'sharpe_ratio',
                'search_space': {
                    'learning_rate': [1e-5, 1e-2],
                    'batch_size': [32, 256],
                    'gamma': [0.9, 0.999],
                    'ent_coef': [0.0, 0.1]
                }
            },
            'early_stopping': {
                'enabled': True,
                'patience': 10,
                'min_delta': 0.01,
                'monitor_metric': 'eval_reward'
            },
            'model_checkpointing': {
                'enabled': True,
                'save_best': True,
                'save_frequency': 10000,
                'max_checkpoints': 5
            },
            'evaluation': {
                'n_eval_episodes': 10,
                'deterministic': True,
                'render': False
            }
        }
    
    def train_agent(self, 
                   agent,
                   env,
                   total_timesteps: int = None,
                   eval_env = None,
                   callbacks: List = None) -> Dict:
        """
        Train the trading agent with comprehensive monitoring.
        
        Args:
            agent: Trading agent to train
            env: Training environment
            total_timesteps: Total training timesteps
            eval_env: Evaluation environment
            callbacks: Training callbacks
            
        Returns:
            Training results dictionary
        """
        logger.info("Starting agent training")
        
        # Set training parameters
        timesteps = total_timesteps or self.config['training']['total_timesteps']
        
        # Set up callbacks
        if callbacks is None:
            callbacks = []
        
        # Add training manager callbacks
        training_callbacks = self._create_training_callbacks()
        callbacks.extend(training_callbacks)
        
        # Add evaluation callback if eval_env provided
        if eval_env is not None:
            eval_callback = self._create_evaluation_callback(eval_env)
            callbacks.append(eval_callback)
        
        # Start training
        start_time = datetime.now()
        
        try:
            # Train the agent
            training_results = agent.train(
                total_timesteps=timesteps,
                eval_env=eval_env,
                callbacks=callbacks
            )
            
            training_time = datetime.now() - start_time
            
            # Record training session
            training_record = {
                'timestamp': start_time,
                'training_time': training_time,
                'total_timesteps': timesteps,
                'results': training_results,
                'config': self.config,
                'success': True
            }
            
            self.training_history.append(training_record)
            
            logger.info(f"Agent training completed in {training_time}")
            return training_results
            
        except Exception as e:
            logger.error(f"Error during agent training: {e}")
            
            # Record failed training session
            training_record = {
                'timestamp': start_time,
                'training_time': datetime.now() - start_time,
                'total_timesteps': timesteps,
                'error': str(e),
                'config': self.config,
                'success': False
            }
            
            self.training_history.append(training_record)
            
            return {
                'success': False,
                'error': str(e),
                'training_time': datetime.now() - start_time
            }
    
    def _create_training_callbacks(self) -> List:
        """Create training manager callbacks."""
        callbacks = []
        
        # Add checkpoint callback if enabled
        if self.config['model_checkpointing']['enabled']:
            checkpoint_callback = self._create_checkpoint_callback()
            callbacks.append(checkpoint_callback)
        
        # Add early stopping callback if enabled
        if self.config['early_stopping']['enabled']:
            early_stopping_callback = self._create_early_stopping_callback()
            callbacks.append(early_stopping_callback)
        
        return callbacks
    
    def _create_checkpoint_callback(self):
        """Create model checkpointing callback."""
        # This would be implemented with stable-baselines3 callbacks
        # For now, return a placeholder
        class CheckpointCallback:
            def __init__(self, manager):
                self.manager = manager
                self.checkpoint_count = 0
            
            def __call__(self, locals_, globals_):
                # Checkpoint logic would go here
                pass
        
        return CheckpointCallback(self)
    
    def _create_early_stopping_callback(self):
        """Create early stopping callback."""
        # This would be implemented with stable-baselines3 callbacks
        # For now, return a placeholder
        class EarlyStoppingCallback:
            def __init__(self, manager):
                self.manager = manager
                self.best_score = -np.inf
                self.patience_counter = 0
            
            def __call__(self, locals_, globals_):
                # Early stopping logic would go here
                pass
        
        return EarlyStoppingCallback(self)
    
    def _create_evaluation_callback(self, eval_env):
        """Create evaluation callback."""
        # This would be implemented with stable-baselines3 callbacks
        # For now, return a placeholder
        class EvaluationCallback:
            def __init__(self, manager, eval_env):
                self.manager = manager
                self.eval_env = eval_env
                self.eval_results = []
            
            def __call__(self, locals_, globals_):
                # Evaluation logic would go here
                pass
        
        return EvaluationCallback(self, eval_env)
    
    def optimize_hyperparameters(self, 
                                agent_class,
                                env,
                                n_trials: int = None,
                                search_space: Dict = None) -> Dict:
        """
        Optimize hyperparameters using Optuna or similar.
        
        Args:
            agent_class: Agent class to optimize
            env: Training environment
            n_trials: Number of optimization trials
            search_space: Hyperparameter search space
            
        Returns:
            Optimization results dictionary
        """
        if not self.config['hyperparameter_optimization']['enabled']:
            logger.info("Hyperparameter optimization disabled")
            return {}
        
        logger.info("Starting hyperparameter optimization")
        
        n_trials = n_trials or self.config['hyperparameter_optimization']['n_trials']
        search_space = search_space or self.config['hyperparameter_optimization']['search_space']
        
        # This would implement actual hyperparameter optimization
        # For now, return a placeholder
        optimization_results = {
            'n_trials': n_trials,
            'search_space': search_space,
            'best_params': {},
            'best_score': -np.inf,
            'trial_results': []
        }
        
        logger.info("Hyperparameter optimization completed")
        return optimization_results
    
    def evaluate_agent(self, 
                      agent,
                      env,
                      n_episodes: int = None,
                      deterministic: bool = None) -> Dict:
        """
        Evaluate the trained agent.
        
        Args:
            agent: Trained agent
            env: Evaluation environment
            n_episodes: Number of evaluation episodes
            deterministic: Whether to use deterministic policy
            
        Returns:
            Evaluation results dictionary
        """
        logger.info("Starting agent evaluation")
        
        n_episodes = n_episodes or self.config['evaluation']['n_eval_episodes']
        deterministic = deterministic if deterministic is not None else self.config['evaluation']['deterministic']
        
        evaluation_results = {
            'n_episodes': n_episodes,
            'deterministic': deterministic,
            'episode_rewards': [],
            'episode_lengths': [],
            'episode_returns': [],
            'episode_volatilities': [],
            'episode_sharpe_ratios': []
        }
        
        for episode in range(n_episodes):
            logger.info(f"Evaluation episode {episode + 1}/{n_episodes}")
            
            obs, _ = env.reset()
            done = False
            episode_reward = 0
            episode_length = 0
            episode_returns = []
            
            while not done:
                # Get action from agent
                action, _ = agent.predict(obs, deterministic=deterministic)
                
                # Execute action
                obs, reward, terminated, truncated, _ = env.step(action)
                done = terminated or truncated
                
                episode_reward += reward
                episode_length += 1
                episode_returns.append(reward)
            
            # Calculate episode metrics
            episode_volatility = np.std(episode_returns) if len(episode_returns) > 1 else 0
            episode_sharpe = np.mean(episode_returns) / episode_volatility if episode_volatility > 0 else 0
            
            evaluation_results['episode_rewards'].append(episode_reward)
            evaluation_results['episode_lengths'].append(episode_length)
            evaluation_results['episode_returns'].append(episode_returns)
            evaluation_results['episode_volatilities'].append(episode_volatility)
            evaluation_results['episode_sharpe_ratios'].append(episode_sharpe)
        
        # Calculate summary statistics
        evaluation_results['mean_reward'] = np.mean(evaluation_results['episode_rewards'])
        evaluation_results['std_reward'] = np.std(evaluation_results['episode_rewards'])
        evaluation_results['mean_length'] = np.mean(evaluation_results['episode_lengths'])
        evaluation_results['mean_sharpe'] = np.mean(evaluation_results['episode_sharpe_ratios'])
        
        logger.info(f"Agent evaluation completed. Mean reward: {evaluation_results['mean_reward']:.2f}")
        return evaluation_results
    
    def get_training_summary(self) -> Dict:
        """Get summary of training history."""
        if not self.training_history:
            return {}
        
        successful_trainings = [t for t in self.training_history if t['success']]
        failed_trainings = [t for t in self.training_history if not t['success']]
        
        summary = {
            'total_trainings': len(self.training_history),
            'successful_trainings': len(successful_trainings),
            'failed_trainings': len(failed_trainings),
            'success_rate': len(successful_trainings) / len(self.training_history) if self.training_history else 0,
            'total_training_time': sum(t['training_time'].total_seconds() for t in successful_trainings),
            'average_training_time': np.mean([t['training_time'].total_seconds() for t in successful_trainings]) if successful_trainings else 0,
            'best_models': list(self.best_models.keys()),
            'hyperparameter_optimizations': len(self.hyperparameter_results)
        }
        
        return summary
    
    def save_training_history(self, path: str):
        """Save training history to file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        # Convert datetime objects to strings for JSON serialization
        serializable_history = []
        for record in self.training_history:
            serializable_record = record.copy()
            serializable_record['timestamp'] = record['timestamp'].isoformat()
            if 'training_time' in record:
                serializable_record['training_time'] = record['training_time'].total_seconds()
            serializable_history.append(serializable_record)
        
        with open(path, 'w') as f:
            json.dump(serializable_history, f, indent=2)
        
        logger.info(f"Training history saved to {path}")
    
    def load_training_history(self, path: str):
        """Load training history from file."""
        path = Path(path)
        
        if not path.exists():
            raise FileNotFoundError(f"Training history file not found: {path}")
        
        with open(path, 'r') as f:
            serializable_history = json.load(f)
        
        # Convert back to datetime objects
        self.training_history = []
        for record in serializable_history:
            record['timestamp'] = datetime.fromisoformat(record['timestamp'])
            if 'training_time' in record:
                record['training_time'] = timedelta(seconds=record['training_time'])
            self.training_history.append(record)
        
        logger.info(f"Training history loaded from {path}")
    
    def get_best_model(self, metric: str = 'sharpe_ratio') -> Optional[Dict]:
        """Get the best model based on specified metric."""
        if metric not in self.best_models:
            return None
        
        return self.best_models[metric]
    
    def update_config(self, new_config: Dict):
        """Update training configuration."""
        self.config.update(new_config)
        logger.info("Training configuration updated")
    
    def get_info(self) -> Dict:
        """Get training manager information."""
        return {
            'config': self.config,
            'training_history_length': len(self.training_history),
            'best_models_count': len(self.best_models),
            'hyperparameter_results_count': len(self.hyperparameter_results)
        }
