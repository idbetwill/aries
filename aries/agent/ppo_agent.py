"""
PPO (Proximal Policy Optimization) Agent Implementation

Implements PPO algorithm for risk-averse energy trading with
stable-baselines3 integration.
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Union, Any
import logging
import warnings
from pathlib import Path
import joblib

try:
    from stable_baselines3 import PPO
    from stable_baselines3.common.env_util import make_vec_env
    from stable_baselines3.common.callbacks import BaseCallback, EvalCallback
    from stable_baselines3.common.vec_env import VecEnv
    SB3_AVAILABLE = True
except ImportError:
    SB3_AVAILABLE = False
    print("Warning: stable-baselines3 not available. Install with: pip install stable-baselines3")

warnings.filterwarnings('ignore')
logger = logging.getLogger(__name__)


class PPOAgent:
    """
    PPO-based trading agent for energy market trading.
    
    Implements Proximal Policy Optimization with risk-averse features
    for energy market trading using stable-baselines3.
    """
    
    def __init__(self, env, config: Dict = None):
        """
        Initialize the PPO agent.
        
        Args:
            env: Trading environment
            config: Configuration dictionary
        """
        if not SB3_AVAILABLE:
            raise ImportError("stable-baselines3 is required for PPO agent")
        
        self.env = env
        self.config = config or self._default_config()
        self.model = None
        self.is_trained = False
        
        # Initialize PPO model
        self._initialize_model()
        
    def _default_config(self) -> Dict:
        """Return default PPO configuration."""
        return {
            'learning_rate': 3e-4,
            'n_steps': 2048,
            'batch_size': 64,
            'n_epochs': 10,
            'gamma': 0.99,
            'gae_lambda': 0.95,
            'clip_range': 0.2,
            'ent_coef': 0.01,
            'vf_coef': 0.5,
            'max_grad_norm': 0.5,
            'target_kl': 0.01,
            'tensorboard_log': None,
            'verbose': 1,
            'device': 'auto',
            'policy': 'MlpPolicy',
            'policy_kwargs': {
                'net_arch': [dict(pi=[256, 256], vf=[256, 256])],
                'activation_fn': 'tanh'
            }
        }
    
    def _initialize_model(self):
        """Initialize the PPO model."""
        logger.info("Initializing PPO model")
        
        # Create PPO model
        self.model = PPO(
            policy=self.config['policy'],
            env=self.env,
            learning_rate=self.config['learning_rate'],
            n_steps=self.config['n_steps'],
            batch_size=self.config['batch_size'],
            n_epochs=self.config['n_epochs'],
            gamma=self.config['gamma'],
            gae_lambda=self.config['gae_lambda'],
            clip_range=self.config['clip_range'],
            ent_coef=self.config['ent_coef'],
            vf_coef=self.config['vf_coef'],
            max_grad_norm=self.config['max_grad_norm'],
            target_kl=self.config['target_kl'],
            tensorboard_log=self.config['tensorboard_log'],
            verbose=self.config['verbose'],
            device=self.config['device'],
            policy_kwargs=self.config['policy_kwargs']
        )
        
        logger.info("PPO model initialized")
    
    def train(self, 
              total_timesteps: int = 100000,
              eval_env = None,
              eval_freq: int = 10000,
              callbacks: List = None) -> Dict:
        """
        Train the PPO agent.
        
        Args:
            total_timesteps: Total number of training timesteps
            eval_env: Evaluation environment
            eval_freq: Evaluation frequency
            callbacks: Training callbacks
            
        Returns:
            Training results dictionary
        """
        if self.model is None:
            raise RuntimeError("PPO model not initialized")
        
        logger.info(f"Starting PPO training for {total_timesteps} timesteps")
        
        # Set up callbacks
        if callbacks is None:
            callbacks = []
        
        # Add evaluation callback if eval_env provided
        if eval_env is not None:
            eval_callback = EvalCallback(
                eval_env,
                best_model_save_path='./models/ppo_best',
                log_path='./logs/ppo_eval',
                eval_freq=eval_freq,
                deterministic=True,
                render=False
            )
            callbacks.append(eval_callback)
        
        # Train the model
        start_time = datetime.now()
        
        try:
            self.model.learn(
                total_timesteps=total_timesteps,
                callback=callbacks if callbacks else None
            )
            
            training_time = datetime.now() - start_time
            self.is_trained = True
            
            results = {
                'training_time': training_time,
                'total_timesteps': total_timesteps,
                'success': True,
                'model_info': self.get_model_info()
            }
            
            logger.info(f"PPO training completed in {training_time}")
            return results
            
        except Exception as e:
            logger.error(f"Error during PPO training: {e}")
            return {
                'training_time': datetime.now() - start_time,
                'total_timesteps': total_timesteps,
                'success': False,
                'error': str(e)
            }
    
    def predict(self, observation: np.ndarray, deterministic: bool = True) -> Tuple[np.ndarray, Dict]:
        """
        Make a prediction using the trained PPO model.
        
        Args:
            observation: Current observation
            deterministic: Whether to use deterministic policy
            
        Returns:
            Tuple of (action, info)
        """
        if not self.is_trained:
            raise RuntimeError("Model must be trained before making predictions")
        
        if self.model is None:
            raise RuntimeError("PPO model not initialized")
        
        try:
            action, _states = self.model.predict(
                observation,
                deterministic=deterministic
            )
            
            info = {
                'action_type': 'ppo',
                'deterministic': deterministic,
                'model_loaded': True
            }
            
            return action, info
            
        except Exception as e:
            logger.error(f"Error in PPO prediction: {e}")
            # Return random action as fallback
            action = self.env.action_space.sample()
            info = {
                'action_type': 'random_fallback',
                'error': str(e)
            }
            return action, info
    
    def get_model_info(self) -> Dict:
        """Get information about the PPO model."""
        if self.model is None:
            return {'model_initialized': False}
        
        info = {
            'model_initialized': True,
            'is_trained': self.is_trained,
            'config': self.config,
            'policy': str(self.model.policy),
            'device': str(self.model.device)
        }
        
        # Add model statistics if available
        if hasattr(self.model, 'get_parameters'):
            try:
                params = self.model.get_parameters()
                info['total_parameters'] = sum(p.numel() for p in params.values() if hasattr(p, 'numel'))
            except:
                pass
        
        return info
    
    def save(self, path: str):
        """Save the trained PPO model."""
        if not self.is_trained:
            raise RuntimeError("Model must be trained before saving")
        
        if self.model is None:
            raise RuntimeError("PPO model not initialized")
        
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        try:
            # Save the model
            self.model.save(str(path))
            
            # Save additional configuration
            config_path = path.parent / f"{path.stem}_config.pkl"
            joblib.dump(self.config, config_path)
            
            logger.info(f"PPO model saved to {path}")
            
        except Exception as e:
            logger.error(f"Error saving PPO model: {e}")
            raise
    
    def load(self, path: str):
        """Load a trained PPO model."""
        path = Path(path)
        
        if not path.exists():
            raise FileNotFoundError(f"PPO model file not found: {path}")
        
        try:
            # Load the model
            self.model = PPO.load(str(path), env=self.env)
            
            # Load configuration if available
            config_path = path.parent / f"{path.stem}_config.pkl"
            if config_path.exists():
                self.config = joblib.load(config_path)
            
            self.is_trained = True
            logger.info(f"PPO model loaded from {path}")
            
        except Exception as e:
            logger.error(f"Error loading PPO model: {e}")
            raise
    
    def get_action_probabilities(self, observation: np.ndarray) -> np.ndarray:
        """
        Get action probabilities from the policy.
        
        Args:
            observation: Current observation
            
        Returns:
            Action probabilities
        """
        if not self.is_trained or self.model is None:
            raise RuntimeError("Model must be trained before getting action probabilities")
        
        try:
            # Get action and state from model
            action, state = self.model.predict(observation, deterministic=False)
            
            # Get policy network
            policy = self.model.policy
            
            # Get action probabilities
            with torch.no_grad():
                obs_tensor = torch.FloatTensor(observation).unsqueeze(0)
                if hasattr(policy, 'get_distribution'):
                    dist = policy.get_distribution(obs_tensor)
                    probs = dist.probs if hasattr(dist, 'probs') else None
                else:
                    probs = None
            
            return probs.numpy() if probs is not None else None
            
        except Exception as e:
            logger.error(f"Error getting action probabilities: {e}")
            return None
    
    def set_learning_rate(self, learning_rate: float):
        """Set the learning rate for the model."""
        if self.model is not None:
            self.model.learning_rate = learning_rate
            logger.info(f"Learning rate set to {learning_rate}")
    
    def get_learning_rate(self) -> float:
        """Get the current learning rate."""
        if self.model is not None:
            return self.model.learning_rate
        return self.config['learning_rate']
    
    def get_training_info(self) -> Dict:
        """Get training information."""
        return {
            'is_trained': self.is_trained,
            'config': self.config,
            'model_info': self.get_model_info()
        }


if SB3_AVAILABLE:
    class PPOTrainingCallback(BaseCallback):
        """
        Custom callback for PPO training monitoring.
        """
        
        def __init__(self, verbose: int = 0):
            super(PPOTrainingCallback, self).__init__(verbose)
            self.training_metrics = []
        
        def _on_step(self) -> bool:
            """Called at each step during training."""
            # Log training metrics
            if self.n_calls % 1000 == 0:
                metrics = {
                    'timestep': self.n_calls,
                    'episode_reward': self.locals.get('rewards', [0])[-1] if 'rewards' in self.locals else 0,
                    'episode_length': self.locals.get('episode_lengths', [0])[-1] if 'episode_lengths' in self.locals else 0
                }
                self.training_metrics.append(metrics)
                
                if self.verbose > 0:
                    logger.info(f"Step {self.n_calls}: Reward = {metrics['episode_reward']:.2f}")
            
            return True
        
        def get_training_metrics(self) -> List[Dict]:
            """Get collected training metrics."""
            return self.training_metrics
else:
    # Dummy class when stable-baselines3 is not available
    class PPOTrainingCallback:
        def __init__(self, verbose: int = 0):
            self.verbose = verbose
            self.training_metrics = []
        
        def _on_step(self) -> bool:
            return True
        
        def get_training_metrics(self) -> List[Dict]:
            return self.training_metrics
