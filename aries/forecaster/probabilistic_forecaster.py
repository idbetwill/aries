"""
Probabilistic Forecaster for Energy Market

Main forecasting class that coordinates different forecasting models
and provides probabilistic predictions with uncertainty quantification.
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Union, Any
import logging
import warnings
from pathlib import Path
import joblib

from .lstm_forecaster import LSTMForecaster
from .transformer_forecaster import TransformerForecaster
from .ensemble_forecaster import EnsembleForecaster
from .evaluation import ForecastEvaluator

warnings.filterwarnings('ignore')
logger = logging.getLogger(__name__)


class ProbabilisticForecaster:
    """
    Main probabilistic forecasting class for energy market predictions.
    
    Coordinates multiple forecasting models and provides ensemble predictions
    with uncertainty quantification for energy prices, demand, and supply.
    """
    
    def __init__(self, config: Dict = None):
        """
        Initialize the probabilistic forecaster.
        
        Args:
            config: Configuration dictionary for forecasting models
        """
        self.config = config or self._default_config()
        self.models = {}
        self.ensemble_forecaster = None
        self.evaluator = ForecastEvaluator()
        self.is_trained = False
        
        # Initialize individual models
        self._initialize_models()
        
    def _default_config(self) -> Dict:
        """Return default forecasting configuration."""
        return {
            'forecast_horizon': 24,  # hours
            'sequence_length': 168,  # hours (1 week)
            'target_variables': ['price', 'demand', 'supply'],
            'features': [
                'price', 'price_change', 'price_volatility_24h',
                'demand', 'supply', 'demand_supply_ratio',
                'hour', 'day_of_week', 'is_weekend',
                'temperature', 'humidity', 'wind_speed'
            ],
            'models': {
                'lstm': {
                    'enabled': True,
                    'hidden_size': 128,
                    'num_layers': 2,
                    'dropout': 0.2,
                    'learning_rate': 0.001,
                    'epochs': 100,
                    'batch_size': 32
                },
                'transformer': {
                    'enabled': True,
                    'd_model': 128,
                    'nhead': 8,
                    'num_layers': 4,
                    'dropout': 0.1,
                    'learning_rate': 0.0001,
                    'epochs': 50,
                    'batch_size': 16
                },
                'ensemble': {
                    'enabled': True,
                    'method': 'weighted_average',  # 'weighted_average', 'stacking', 'voting'
                    'weights': [0.4, 0.6]  # LSTM, Transformer
                }
            },
            'uncertainty_quantification': {
                'method': 'quantile_regression',  # 'quantile_regression', 'monte_carlo', 'ensemble'
                'quantiles': [0.05, 0.25, 0.5, 0.75, 0.95],
                'monte_carlo_samples': 100
            },
            'validation': {
                'train_split': 0.7,
                'val_split': 0.15,
                'test_split': 0.15,
                'cross_validation': True,
                'cv_folds': 5
            }
        }
    
    def _initialize_models(self):
        """Initialize individual forecasting models."""
        logger.info("Initializing forecasting models")
        
        # LSTM Forecaster
        if self.config['models']['lstm']['enabled']:
            lstm_config = self.config['models']['lstm'].copy()
            lstm_config.update({
                'sequence_length': self.config['sequence_length'],
                'forecast_horizon': self.config['forecast_horizon'],
                'target_variables': self.config['target_variables'],
                'features': self.config['features']
            })
            self.models['lstm'] = LSTMForecaster(config=lstm_config)
        
        # Transformer Forecaster
        if self.config['models']['transformer']['enabled']:
            transformer_config = self.config['models']['transformer'].copy()
            transformer_config.update({
                'sequence_length': self.config['sequence_length'],
                'forecast_horizon': self.config['forecast_horizon'],
                'target_variables': self.config['target_variables'],
                'features': self.config['features']
            })
            self.models['transformer'] = TransformerForecaster(config=transformer_config)
        
        # Ensemble Forecaster
        if self.config['models']['ensemble']['enabled']:
            ensemble_config = self.config['models']['ensemble'].copy()
            ensemble_config.update({
                'forecast_horizon': self.config['forecast_horizon'],
                'target_variables': self.config['target_variables']
            })
            self.ensemble_forecaster = EnsembleForecaster(config=ensemble_config)
    
    def prepare_data(self, data: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """
        Prepare data for forecasting models.
        
        Args:
            data: Input data DataFrame
            
        Returns:
            Tuple of (X, y, feature_names)
        """
        logger.info("Preparing data for forecasting")
        
        # Select features
        feature_names = self.config['features']
        available_features = [f for f in feature_names if f in data.columns]
        
        if not available_features:
            raise ValueError("No valid features found in data")
        
        # Prepare feature matrix
        X = data[available_features].values
        
        # Prepare target variables
        target_variables = self.config['target_variables']
        available_targets = [t for t in target_variables if t in data.columns]
        
        if not available_targets:
            raise ValueError("No valid target variables found in data")
        
        y = data[available_targets].values
        
        # Handle missing values
        X = np.nan_to_num(X, nan=0.0)
        y = np.nan_to_num(y, nan=0.0)
        
        logger.info(f"Prepared data: {X.shape} features, {y.shape} targets")
        return X, y, available_features
    
    def train(self, data: pd.DataFrame, validation_data: pd.DataFrame = None) -> Dict:
        """
        Train all forecasting models.
        
        Args:
            data: Training data DataFrame
            validation_data: Validation data DataFrame
            
        Returns:
            Training results dictionary
        """
        logger.info("Starting model training")
        
        # Prepare data
        X, y, feature_names = self.prepare_data(data)
        
        # Split data if validation data not provided
        if validation_data is None:
            train_size = int(len(X) * self.config['validation']['train_split'])
            val_size = int(len(X) * self.config['validation']['val_split'])
            
            X_train = X[:train_size]
            y_train = y[:train_size]
            X_val = X[train_size:train_size + val_size]
            y_val = y[train_size:train_size + val_size]
        else:
            X_train, y_train, _ = self.prepare_data(data)
            X_val, y_val, _ = self.prepare_data(validation_data)
        
        training_results = {}
        
        # Train individual models
        for model_name, model in self.models.items():
            logger.info(f"Training {model_name} model")
            
            try:
                result = model.train(X_train, y_train, X_val, y_val)
                training_results[model_name] = result
                logger.info(f"{model_name} training completed successfully")
            except Exception as e:
                logger.error(f"Error training {model_name}: {e}")
                training_results[model_name] = {'error': str(e)}
        
        # Train ensemble if enabled
        if self.ensemble_forecaster and len(self.models) > 1:
            logger.info("Training ensemble model")
            
            try:
                ensemble_result = self.ensemble_forecaster.train(
                    self.models, X_train, y_train, X_val, y_val
                )
                training_results['ensemble'] = ensemble_result
                logger.info("Ensemble training completed successfully")
            except Exception as e:
                logger.error(f"Error training ensemble: {e}")
                training_results['ensemble'] = {'error': str(e)}
        
        self.is_trained = True
        logger.info("All model training completed")
        
        return training_results
    
    def predict(self, data: pd.DataFrame, horizon: int = None) -> Dict:
        """
        Make probabilistic predictions.
        
        Args:
            data: Input data for prediction
            horizon: Forecast horizon (if None, uses config default)
            
        Returns:
            Dictionary with predictions and uncertainty
        """
        if not self.is_trained:
            raise RuntimeError("Models must be trained before making predictions")
        
        horizon = horizon or self.config['forecast_horizon']
        logger.info(f"Making predictions for {horizon} hours ahead")
        
        # Prepare input data
        X, _, feature_names = self.prepare_data(data)
        
        # Get predictions from individual models
        model_predictions = {}
        
        for model_name, model in self.models.items():
            try:
                pred = model.predict(X, horizon=horizon)
                model_predictions[model_name] = pred
            except Exception as e:
                logger.error(f"Error getting predictions from {model_name}: {e}")
        
        # Get ensemble predictions
        if self.ensemble_forecaster and len(model_predictions) > 1:
            try:
                ensemble_pred = self.ensemble_forecaster.predict(
                    model_predictions, horizon=horizon
                )
                model_predictions['ensemble'] = ensemble_pred
            except Exception as e:
                logger.error(f"Error getting ensemble predictions: {e}")
        
        # Quantify uncertainty
        uncertainty = self._quantify_uncertainty(model_predictions)
        
        # Combine predictions
        final_predictions = self._combine_predictions(model_predictions, uncertainty)
        
        return final_predictions
    
    def _quantify_uncertainty(self, model_predictions: Dict) -> Dict:
        """Quantify prediction uncertainty."""
        method = self.config['uncertainty_quantification']['method']
        
        if method == 'quantile_regression':
            return self._quantile_uncertainty(model_predictions)
        elif method == 'monte_carlo':
            return self._monte_carlo_uncertainty(model_predictions)
        elif method == 'ensemble':
            return self._ensemble_uncertainty(model_predictions)
        else:
            logger.warning(f"Unknown uncertainty method: {method}")
            return {}
    
    def _quantile_uncertainty(self, model_predictions: Dict) -> Dict:
        """Calculate uncertainty using quantile regression."""
        quantiles = self.config['uncertainty_quantification']['quantiles']
        uncertainty = {}
        
        for target in self.config['target_variables']:
            target_predictions = []
            
            for model_name, predictions in model_predictions.items():
                if target in predictions:
                    target_predictions.append(predictions[target])
            
            if target_predictions:
                # Calculate quantiles across models
                all_predictions = np.concatenate(target_predictions, axis=0)
                quantile_values = np.percentile(all_predictions, 
                                              [q * 100 for q in quantiles], 
                                              axis=0)
                
                uncertainty[target] = {
                    'quantiles': quantile_values,
                    'mean': np.mean(all_predictions, axis=0),
                    'std': np.std(all_predictions, axis=0)
                }
        
        return uncertainty
    
    def _monte_carlo_uncertainty(self, model_predictions: Dict) -> Dict:
        """Calculate uncertainty using Monte Carlo sampling."""
        n_samples = self.config['uncertainty_quantification']['monte_carlo_samples']
        uncertainty = {}
        
        for target in self.config['target_variables']:
            target_predictions = []
            
            for model_name, predictions in model_predictions.items():
                if target in predictions:
                    target_predictions.append(predictions[target])
            
            if target_predictions:
                # Sample from predictions
                samples = []
                for _ in range(n_samples):
                    model_idx = np.random.randint(0, len(target_predictions))
                    sample = target_predictions[model_idx]
                    samples.append(sample)
                
                samples = np.array(samples)
                
                uncertainty[target] = {
                    'samples': samples,
                    'mean': np.mean(samples, axis=0),
                    'std': np.std(samples, axis=0),
                    'confidence_interval': np.percentile(samples, [5, 95], axis=0)
                }
        
        return uncertainty
    
    def _ensemble_uncertainty(self, model_predictions: Dict) -> Dict:
        """Calculate uncertainty using ensemble variance."""
        uncertainty = {}
        
        for target in self.config['target_variables']:
            target_predictions = []
            
            for model_name, predictions in model_predictions.items():
                if target in predictions:
                    target_predictions.append(predictions[target])
            
            if target_predictions:
                predictions_array = np.array(target_predictions)
                
                uncertainty[target] = {
                    'mean': np.mean(predictions_array, axis=0),
                    'std': np.std(predictions_array, axis=0),
                    'variance': np.var(predictions_array, axis=0)
                }
        
        return uncertainty
    
    def _combine_predictions(self, model_predictions: Dict, uncertainty: Dict) -> Dict:
        """Combine predictions from different models."""
        combined = {}
        
        for target in self.config['target_variables']:
            target_predictions = []
            
            for model_name, predictions in model_predictions.items():
                if target in predictions:
                    target_predictions.append(predictions[target])
            
            if target_predictions:
                # Weighted average of predictions
                weights = self.config['models']['ensemble']['weights']
                if len(weights) == len(target_predictions):
                    weighted_pred = np.average(target_predictions, 
                                             weights=weights, axis=0)
                else:
                    weighted_pred = np.mean(target_predictions, axis=0)
                
                combined[target] = {
                    'prediction': weighted_pred,
                    'uncertainty': uncertainty.get(target, {}),
                    'model_predictions': {name: pred[target] for name, pred in model_predictions.items() 
                                        if target in pred}
                }
        
        return combined
    
    def evaluate(self, test_data: pd.DataFrame, predictions: Dict = None) -> Dict:
        """
        Evaluate forecasting performance.
        
        Args:
            test_data: Test data DataFrame
            predictions: Predictions dictionary (if None, will make predictions)
            
        Returns:
            Evaluation results dictionary
        """
        if predictions is None:
            predictions = self.predict(test_data)
        
        return self.evaluator.evaluate(test_data, predictions)
    
    def save_models(self, path: str):
        """Save trained models to disk."""
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        
        # Save individual models
        for model_name, model in self.models.items():
            model_path = path / f"{model_name}_model.pkl"
            model.save(str(model_path))
        
        # Save ensemble model
        if self.ensemble_forecaster:
            ensemble_path = path / "ensemble_model.pkl"
            self.ensemble_forecaster.save(str(ensemble_path))
        
        # Save configuration
        config_path = path / "config.pkl"
        joblib.dump(self.config, config_path)
        
        logger.info(f"Models saved to {path}")
    
    def load_models(self, path: str):
        """Load trained models from disk."""
        path = Path(path)
        
        # Load configuration
        config_path = path / "config.pkl"
        if config_path.exists():
            self.config = joblib.load(config_path)
        
        # Load individual models
        for model_name in self.models.keys():
            model_path = path / f"{model_name}_model.pkl"
            if model_path.exists():
                self.models[model_name].load(str(model_path))
        
        # Load ensemble model
        if self.ensemble_forecaster:
            ensemble_path = path / "ensemble_model.pkl"
            if ensemble_path.exists():
                self.ensemble_forecaster.load(str(ensemble_path))
        
        self.is_trained = True
        logger.info(f"Models loaded from {path}")
    
    def get_model_info(self) -> Dict:
        """Get information about trained models."""
        info = {
            'is_trained': self.is_trained,
            'models': list(self.models.keys()),
            'ensemble_enabled': self.ensemble_forecaster is not None,
            'config': self.config
        }
        
        # Add model-specific information
        for model_name, model in self.models.items():
            if hasattr(model, 'get_model_info'):
                info[f'{model_name}_info'] = model.get_model_info()
        
        return info
