"""
Ensemble Forecaster for Energy Market Predictions

Combines multiple forecasting models to provide robust predictions
with improved accuracy and uncertainty quantification.
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Union, Any
import logging
import warnings
from pathlib import Path
import joblib
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error

warnings.filterwarnings('ignore')
logger = logging.getLogger(__name__)


class EnsembleForecaster:
    """
    Ensemble forecaster that combines multiple models for robust predictions.
    
    Implements various ensemble methods including weighted averaging,
    stacking, and voting for energy market forecasting.
    """
    
    def __init__(self, config: Dict = None):
        """
        Initialize the ensemble forecaster.
        
        Args:
            config: Configuration dictionary for ensemble methods
        """
        self.config = config or self._default_config()
        self.meta_models = {}
        self.weights = None
        self.is_trained = False
        
    def _default_config(self) -> Dict:
        """Return default ensemble configuration."""
        return {
            'forecast_horizon': 24,
            'target_variables': ['price', 'demand', 'supply'],
            'method': 'weighted_average',  # 'weighted_average', 'stacking', 'voting'
            'weights': [0.4, 0.6],  # Weights for each model
            'stacking_models': {
                'linear': LinearRegression(),
                'rf': RandomForestRegressor(n_estimators=100, random_state=42),
                'gb': GradientBoostingRegressor(n_estimators=100, random_state=42)
            },
            'cross_validation': True,
            'cv_folds': 5,
            'uncertainty_method': 'ensemble_variance'  # 'ensemble_variance', 'quantile_ensemble'
        }
    
    def train(self, models: Dict, X_train: np.ndarray, y_train: np.ndarray,
              X_val: np.ndarray = None, y_val: np.ndarray = None) -> Dict:
        """
        Train the ensemble forecaster.
        
        Args:
            models: Dictionary of trained models
            X_train: Training input features
            y_train: Training target variables
            X_val: Validation input features
            y_val: Validation target variables
            
        Returns:
            Training results dictionary
        """
        logger.info("Starting ensemble training")
        
        method = self.config['method']
        
        if method == 'weighted_average':
            return self._train_weighted_average(models, X_train, y_train, X_val, y_val)
        elif method == 'stacking':
            return self._train_stacking(models, X_train, y_train, X_val, y_val)
        elif method == 'voting':
            return self._train_voting(models, X_train, y_train, X_val, y_val)
        else:
            raise ValueError(f"Unknown ensemble method: {method}")
    
    def _train_weighted_average(self, models: Dict, X_train: np.ndarray, y_train: np.ndarray,
                               X_val: np.ndarray = None, y_val: np.ndarray = None) -> Dict:
        """Train weighted average ensemble."""
        logger.info("Training weighted average ensemble")
        
        # Get predictions from all models
        model_predictions = {}
        model_errors = {}
        
        for model_name, model in models.items():
            try:
                # Get predictions
                pred = model.predict(X_train)
                
                # Calculate error for weight calculation
                if X_val is not None and y_val is not None:
                    val_pred = model.predict(X_val)
                    error = mean_squared_error(y_val, val_pred)
                else:
                    error = mean_squared_error(y_train, pred)
                
                model_predictions[model_name] = pred
                model_errors[model_name] = error
                
            except Exception as e:
                logger.error(f"Error getting predictions from {model_name}: {e}")
                continue
        
        if not model_predictions:
            raise RuntimeError("No valid model predictions available")
        
        # Calculate weights based on inverse error
        errors = np.array(list(model_errors.values()))
        weights = 1.0 / (errors + 1e-8)  # Add small value to avoid division by zero
        weights = weights / np.sum(weights)  # Normalize weights
        
        self.weights = dict(zip(model_predictions.keys(), weights))
        self.is_trained = True
        
        logger.info(f"Ensemble weights: {self.weights}")
        
        return {
            'method': 'weighted_average',
            'weights': self.weights,
            'model_errors': model_errors
        }
    
    def _train_stacking(self, models: Dict, X_train: np.ndarray, y_train: np.ndarray,
                       X_val: np.ndarray = None, y_val: np.ndarray = None) -> Dict:
        """Train stacking ensemble."""
        logger.info("Training stacking ensemble")
        
        # Get base model predictions
        base_predictions = []
        model_names = []
        
        for model_name, model in models.items():
            try:
                pred = model.predict(X_train)
                base_predictions.append(pred)
                model_names.append(model_name)
            except Exception as e:
                logger.error(f"Error getting predictions from {model_name}: {e}")
                continue
        
        if not base_predictions:
            raise RuntimeError("No valid model predictions available")
        
        # Stack predictions
        X_stacked = np.column_stack(base_predictions)
        
        # Train meta-models for each target variable
        meta_models = {}
        
        for i, target in enumerate(self.config['target_variables']):
            if i < y_train.shape[1]:
                y_target = y_train[:, i]
                
                # Train multiple meta-models
                target_meta_models = {}
                
                for meta_name, meta_model in self.config['stacking_models'].items():
                    try:
                        meta_model.fit(X_stacked, y_target)
                        target_meta_models[meta_name] = meta_model
                    except Exception as e:
                        logger.error(f"Error training meta-model {meta_name} for {target}: {e}")
                
                meta_models[target] = target_meta_models
        
        self.meta_models = meta_models
        self.model_names = model_names
        self.is_trained = True
        
        logger.info("Stacking ensemble training completed")
        
        return {
            'method': 'stacking',
            'meta_models': {target: list(models.keys()) for target, models in meta_models.items()},
            'base_models': model_names
        }
    
    def _train_voting(self, models: Dict, X_train: np.ndarray, y_train: np.ndarray,
                     X_val: np.ndarray = None, y_val: np.ndarray = None) -> Dict:
        """Train voting ensemble."""
        logger.info("Training voting ensemble")
        
        # For voting, we just store the models
        self.models = models
        self.is_trained = True
        
        logger.info("Voting ensemble training completed")
        
        return {
            'method': 'voting',
            'models': list(models.keys())
        }
    
    def predict(self, model_predictions: Dict, horizon: int = None) -> Dict:
        """
        Make ensemble predictions.
        
        Args:
            model_predictions: Dictionary of predictions from individual models
            horizon: Forecast horizon (if None, uses config default)
            
        Returns:
            Dictionary with ensemble predictions
        """
        if not self.is_trained:
            raise RuntimeError("Ensemble must be trained before making predictions")
        
        horizon = horizon or self.config['forecast_horizon']
        method = self.config['method']
        
        if method == 'weighted_average':
            return self._predict_weighted_average(model_predictions, horizon)
        elif method == 'stacking':
            return self._predict_stacking(model_predictions, horizon)
        elif method == 'voting':
            return self._predict_voting(model_predictions, horizon)
        else:
            raise ValueError(f"Unknown ensemble method: {method}")
    
    def _predict_weighted_average(self, model_predictions: Dict, horizon: int) -> Dict:
        """Make weighted average predictions."""
        ensemble_predictions = {}
        
        for target in self.config['target_variables']:
            target_predictions = []
            weights = []
            
            for model_name, predictions in model_predictions.items():
                if target in predictions and model_name in self.weights:
                    target_predictions.append(predictions[target])
                    weights.append(self.weights[model_name])
            
            if target_predictions:
                # Weighted average
                predictions_array = np.array(target_predictions)
                weights_array = np.array(weights)
                weights_array = weights_array / np.sum(weights_array)  # Normalize
                
                weighted_pred = np.average(predictions_array, weights=weights_array, axis=0)
                
                # Calculate uncertainty
                uncertainty = self._calculate_uncertainty(predictions_array, weights_array)
                
                ensemble_predictions[target] = {
                    'prediction': weighted_pred,
                    'uncertainty': uncertainty,
                    'model_predictions': {name: pred[target] for name, pred in model_predictions.items() 
                                        if target in pred}
                }
        
        return ensemble_predictions
    
    def _predict_stacking(self, model_predictions: Dict, horizon: int) -> Dict:
        """Make stacking predictions."""
        ensemble_predictions = {}
        
        # Prepare stacked features
        stacked_features = []
        model_names = []
        
        for model_name, predictions in model_predictions.items():
            # Extract predictions for stacking
            model_pred = []
            for target in self.config['target_variables']:
                if target in predictions:
                    pred = predictions[target]
                    if isinstance(pred, dict) and 'mean' in pred:
                        model_pred.extend(pred['mean'])
                    elif isinstance(pred, np.ndarray):
                        model_pred.extend(pred.flatten())
                    else:
                        model_pred.extend([0.0] * horizon)
            
            stacked_features.append(model_pred)
            model_names.append(model_name)
        
        if not stacked_features:
            logger.warning("No valid predictions for stacking")
            return {}
        
        X_stacked = np.array(stacked_features).T
        
        # Make predictions for each target
        for i, target in enumerate(self.config['target_variables']):
            if target in self.meta_models:
                target_meta_models = self.meta_models[target]
                
                # Average predictions from all meta-models
                meta_predictions = []
                for meta_name, meta_model in target_meta_models.items():
                    try:
                        pred = meta_model.predict(X_stacked)
                        meta_predictions.append(pred)
                    except Exception as e:
                        logger.error(f"Error in meta-model {meta_name} for {target}: {e}")
                
                if meta_predictions:
                    ensemble_pred = np.mean(meta_predictions, axis=0)
                    uncertainty = np.std(meta_predictions, axis=0)
                    
                    ensemble_predictions[target] = {
                        'prediction': ensemble_pred,
                        'uncertainty': uncertainty,
                        'meta_predictions': {name: pred for name, pred in zip(target_meta_models.keys(), meta_predictions)}
                    }
        
        return ensemble_predictions
    
    def _predict_voting(self, model_predictions: Dict, horizon: int) -> Dict:
        """Make voting predictions."""
        ensemble_predictions = {}
        
        for target in self.config['target_variables']:
            target_predictions = []
            
            for model_name, predictions in model_predictions.items():
                if target in predictions:
                    pred = predictions[target]
                    if isinstance(pred, dict) and 'mean' in pred:
                        target_predictions.append(pred['mean'])
                    elif isinstance(pred, np.ndarray):
                        target_predictions.append(pred)
            
            if target_predictions:
                # Simple voting (average)
                predictions_array = np.array(target_predictions)
                voting_pred = np.mean(predictions_array, axis=0)
                uncertainty = np.std(predictions_array, axis=0)
                
                ensemble_predictions[target] = {
                    'prediction': voting_pred,
                    'uncertainty': uncertainty,
                    'model_predictions': {name: pred[target] for name, pred in model_predictions.items() 
                                        if target in pred}
                }
        
        return ensemble_predictions
    
    def _calculate_uncertainty(self, predictions: np.ndarray, weights: np.ndarray) -> Dict:
        """Calculate uncertainty from ensemble predictions."""
        uncertainty_method = self.config['uncertainty_method']
        
        if uncertainty_method == 'ensemble_variance':
            # Weighted variance
            mean_pred = np.average(predictions, weights=weights, axis=0)
            variance = np.average((predictions - mean_pred) ** 2, weights=weights, axis=0)
            
            return {
                'variance': variance,
                'std': np.sqrt(variance),
                'confidence_interval': np.percentile(predictions, [5, 95], axis=0)
            }
        
        elif uncertainty_method == 'quantile_ensemble':
            # Quantile-based uncertainty
            quantiles = np.percentile(predictions, [5, 25, 50, 75, 95], axis=0)
            
            return {
                'quantiles': {
                    0.05: quantiles[0],
                    0.25: quantiles[1],
                    0.50: quantiles[2],
                    0.75: quantiles[3],
                    0.95: quantiles[4]
                },
                'mean': np.mean(predictions, axis=0),
                'std': np.std(predictions, axis=0)
            }
        
        else:
            # Default: simple standard deviation
            return {
                'std': np.std(predictions, axis=0),
                'mean': np.mean(predictions, axis=0)
            }
    
    def save(self, path: str):
        """Save the ensemble model."""
        if not self.is_trained:
            raise RuntimeError("Ensemble must be trained before saving")
        
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        # Save ensemble state
        ensemble_state = {
            'config': self.config,
            'weights': self.weights,
            'meta_models': self.meta_models,
            'model_names': getattr(self, 'model_names', None),
            'is_trained': self.is_trained
        }
        
        joblib.dump(ensemble_state, path)
        logger.info(f"Ensemble model saved to {path}")
    
    def load(self, path: str):
        """Load the ensemble model."""
        path = Path(path)
        
        if not path.exists():
            raise FileNotFoundError(f"Ensemble model file not found: {path}")
        
        # Load ensemble state
        ensemble_state = joblib.load(path)
        
        self.config = ensemble_state['config']
        self.weights = ensemble_state['weights']
        self.meta_models = ensemble_state['meta_models']
        self.model_names = ensemble_state.get('model_names')
        self.is_trained = ensemble_state['is_trained']
        
        logger.info(f"Ensemble model loaded from {path}")
    
    def get_model_info(self) -> Dict:
        """Get information about the ensemble model."""
        return {
            'is_trained': self.is_trained,
            'method': self.config['method'],
            'weights': self.weights,
            'meta_models': list(self.meta_models.keys()) if self.meta_models else None,
            'config': self.config
        }
