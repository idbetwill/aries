"""
Forecast Evaluation Module

Provides comprehensive evaluation metrics for probabilistic forecasting
models including accuracy, uncertainty, and risk metrics.
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Union, Any
import logging
import warnings
from scipy import stats
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

warnings.filterwarnings('ignore')
logger = logging.getLogger(__name__)


class ForecastEvaluator:
    """
    Comprehensive evaluator for probabilistic forecasting models.
    
    Provides various metrics for evaluating forecast accuracy, uncertainty
    quantification, and risk assessment.
    """
    
    def __init__(self, config: Dict = None):
        """
        Initialize the forecast evaluator.
        
        Args:
            config: Configuration dictionary for evaluation metrics
        """
        self.config = config or self._default_config()
        
    def _default_config(self) -> Dict:
        """Return default evaluation configuration."""
        return {
            'metrics': {
                'accuracy': ['mse', 'rmse', 'mae', 'mape', 'r2'],
                'uncertainty': ['crps', 'pinball_loss', 'coverage'],
                'risk': ['var', 'cvar', 'expected_shortfall']
            },
            'quantiles': [0.05, 0.25, 0.5, 0.75, 0.95],
            'confidence_levels': [0.8, 0.9, 0.95],
            'risk_levels': [0.05, 0.1, 0.2]
        }
    
    def evaluate(self, test_data: pd.DataFrame, predictions: Dict) -> Dict:
        """
        Evaluate forecasting performance.
        
        Args:
            test_data: Test data DataFrame
            predictions: Predictions dictionary
            
        Returns:
            Evaluation results dictionary
        """
        logger.info("Starting forecast evaluation")
        
        results = {}
        
        # Extract actual values
        actual_values = self._extract_actual_values(test_data)
        
        # Evaluate each target variable
        for target in self.config.get('target_variables', ['price', 'demand', 'supply']):
            if target in actual_values and target in predictions:
                target_results = self._evaluate_target(
                    actual_values[target],
                    predictions[target]
                )
                results[target] = target_results
        
        # Overall evaluation
        results['overall'] = self._calculate_overall_metrics(results)
        
        logger.info("Forecast evaluation completed")
        return results
    
    def _extract_actual_values(self, test_data: pd.DataFrame) -> Dict:
        """Extract actual values from test data."""
        actual_values = {}
        
        target_variables = self.config.get('target_variables', ['price', 'demand', 'supply'])
        
        for target in target_variables:
            if target in test_data.columns:
                actual_values[target] = test_data[target].values
            else:
                logger.warning(f"Target variable {target} not found in test data")
        
        return actual_values
    
    def _evaluate_target(self, actual: np.ndarray, predictions: Dict) -> Dict:
        """Evaluate predictions for a single target variable."""
        results = {}
        
        # Extract prediction values
        if 'prediction' in predictions:
            pred_values = predictions['prediction']
        elif 'mean' in predictions:
            pred_values = predictions['mean']
        else:
            logger.warning("No prediction values found")
            return {}
        
        # Ensure same length
        min_length = min(len(actual), len(pred_values))
        actual = actual[:min_length]
        pred_values = pred_values[:min_length]
        
        # Accuracy metrics
        results['accuracy'] = self._calculate_accuracy_metrics(actual, pred_values)
        
        # Uncertainty metrics
        if 'uncertainty' in predictions or 'quantiles' in predictions:
            results['uncertainty'] = self._calculate_uncertainty_metrics(
                actual, predictions
            )
        
        # Risk metrics
        results['risk'] = self._calculate_risk_metrics(actual, pred_values)
        
        return results
    
    def _calculate_accuracy_metrics(self, actual: np.ndarray, predicted: np.ndarray) -> Dict:
        """Calculate accuracy metrics."""
        metrics = {}
        
        # Remove NaN values
        mask = ~(np.isnan(actual) | np.isnan(predicted))
        actual_clean = actual[mask]
        predicted_clean = predicted[mask]
        
        if len(actual_clean) == 0:
            return {}
        
        # Mean Squared Error
        metrics['mse'] = mean_squared_error(actual_clean, predicted_clean)
        
        # Root Mean Squared Error
        metrics['rmse'] = np.sqrt(metrics['mse'])
        
        # Mean Absolute Error
        metrics['mae'] = mean_absolute_error(actual_clean, predicted_clean)
        
        # Mean Absolute Percentage Error
        mape = np.mean(np.abs((actual_clean - predicted_clean) / actual_clean)) * 100
        metrics['mape'] = mape
        
        # R-squared
        metrics['r2'] = r2_score(actual_clean, predicted_clean)
        
        # Mean Bias Error
        metrics['mbe'] = np.mean(predicted_clean - actual_clean)
        
        # Normalized Root Mean Squared Error
        metrics['nrmse'] = metrics['rmse'] / np.mean(actual_clean)
        
        return metrics
    
    def _calculate_uncertainty_metrics(self, actual: np.ndarray, predictions: Dict) -> Dict:
        """Calculate uncertainty quantification metrics."""
        metrics = {}
        
        # Continuous Ranked Probability Score (CRPS)
        if 'quantiles' in predictions:
            crps = self._calculate_crps(actual, predictions['quantiles'])
            metrics['crps'] = crps
        
        # Pinball Loss
        if 'quantiles' in predictions:
            pinball_loss = self._calculate_pinball_loss(actual, predictions['quantiles'])
            metrics['pinball_loss'] = pinball_loss
        
        # Coverage
        if 'uncertainty' in predictions and 'confidence_interval' in predictions['uncertainty']:
            coverage = self._calculate_coverage(actual, predictions['uncertainty']['confidence_interval'])
            metrics['coverage'] = coverage
        
        # Sharpness
        if 'uncertainty' in predictions and 'std' in predictions['uncertainty']:
            sharpness = np.mean(predictions['uncertainty']['std'])
            metrics['sharpness'] = sharpness
        
        return metrics
    
    def _calculate_crps(self, actual: np.ndarray, quantiles: Dict) -> float:
        """Calculate Continuous Ranked Probability Score."""
        try:
            # Convert quantiles to distribution parameters
            quantile_values = list(quantiles.values())
            quantile_probs = list(quantiles.keys())
            
            # Simple CRPS calculation using quantiles
            crps_values = []
            for i, obs in enumerate(actual):
                if i < len(quantile_values[0]):
                    # Calculate CRPS for this observation
                    obs_quantiles = [q[i] for q in quantile_values]
                    crps = self._crps_quantile(obs, obs_quantiles, quantile_probs)
                    crps_values.append(crps)
            
            return np.mean(crps_values) if crps_values else np.nan
            
        except Exception as e:
            logger.error(f"Error calculating CRPS: {e}")
            return np.nan
    
    def _crps_quantile(self, obs: float, quantiles: List[float], probs: List[float]) -> float:
        """Calculate CRPS for a single observation using quantiles."""
        # Sort quantiles and probabilities
        sorted_data = sorted(zip(quantiles, probs))
        quantiles_sorted, probs_sorted = zip(*sorted_data)
        
        # Calculate CRPS
        crps = 0.0
        for i in range(len(quantiles_sorted)):
            if obs <= quantiles_sorted[i]:
                crps += (probs_sorted[i] ** 2) * (quantiles_sorted[i] - obs)
            else:
                crps += ((1 - probs_sorted[i]) ** 2) * (obs - quantiles_sorted[i])
        
        return crps
    
    def _calculate_pinball_loss(self, actual: np.ndarray, quantiles: Dict) -> Dict:
        """Calculate pinball loss for each quantile."""
        pinball_losses = {}
        
        for quantile, values in quantiles.items():
            if len(values) == len(actual):
                loss = np.mean(np.maximum(
                    quantile * (actual - values),
                    (quantile - 1) * (actual - values)
                ))
                pinball_losses[quantile] = loss
        
        return pinball_losses
    
    def _calculate_coverage(self, actual: np.ndarray, confidence_interval: np.ndarray) -> Dict:
        """Calculate coverage of confidence intervals."""
        if confidence_interval.shape[0] != 2:
            return {}
        
        lower, upper = confidence_interval[0], confidence_interval[1]
        
        # Calculate coverage for each confidence level
        coverage = {}
        for level in self.config['confidence_levels']:
            alpha = 1 - level
            lower_bound = np.percentile([lower, upper], alpha/2 * 100)
            upper_bound = np.percentile([lower, upper], (1 - alpha/2) * 100)
            
            covered = np.sum((actual >= lower_bound) & (actual <= upper_bound))
            coverage[f'coverage_{level}'] = covered / len(actual)
        
        return coverage
    
    def _calculate_risk_metrics(self, actual: np.ndarray, predicted: np.ndarray) -> Dict:
        """Calculate risk metrics."""
        metrics = {}
        
        # Calculate returns/errors
        errors = predicted - actual
        
        # Value at Risk (VaR)
        for level in self.config['risk_levels']:
            var = np.percentile(errors, level * 100)
            metrics[f'var_{level}'] = var
        
        # Conditional Value at Risk (CVaR)
        for level in self.config['risk_levels']:
            var = np.percentile(errors, level * 100)
            tail_errors = errors[errors <= var]
            cvar = np.mean(tail_errors) if len(tail_errors) > 0 else var
            metrics[f'cvar_{level}'] = cvar
        
        # Expected Shortfall
        for level in self.config['risk_levels']:
            var = np.percentile(errors, level * 100)
            tail_errors = errors[errors <= var]
            es = np.mean(tail_errors) if len(tail_errors) > 0 else var
            metrics[f'es_{level}'] = es
        
        # Maximum Drawdown
        cumulative_errors = np.cumsum(errors)
        running_max = np.maximum.accumulate(cumulative_errors)
        drawdown = cumulative_errors - running_max
        metrics['max_drawdown'] = np.min(drawdown)
        
        return metrics
    
    def _calculate_overall_metrics(self, results: Dict) -> Dict:
        """Calculate overall evaluation metrics."""
        overall = {}
        
        # Aggregate accuracy metrics
        accuracy_metrics = ['mse', 'rmse', 'mae', 'mape', 'r2']
        for metric in accuracy_metrics:
            values = []
            for target, target_results in results.items():
                if 'accuracy' in target_results and metric in target_results['accuracy']:
                    values.append(target_results['accuracy'][metric])
            
            if values:
                overall[f'mean_{metric}'] = np.mean(values)
                overall[f'std_{metric}'] = np.std(values)
        
        # Aggregate uncertainty metrics
        uncertainty_metrics = ['crps', 'sharpness']
        for metric in uncertainty_metrics:
            values = []
            for target, target_results in results.items():
                if 'uncertainty' in target_results and metric in target_results['uncertainty']:
                    values.append(target_results['uncertainty'][metric])
            
            if values:
                overall[f'mean_{metric}'] = np.mean(values)
                overall[f'std_{metric}'] = np.std(values)
        
        return overall
    
    def generate_report(self, results: Dict) -> str:
        """Generate a comprehensive evaluation report."""
        report = []
        report.append("=" * 60)
        report.append("FORECAST EVALUATION REPORT")
        report.append("=" * 60)
        
        # Overall metrics
        if 'overall' in results:
            report.append("\nOVERALL PERFORMANCE:")
            report.append("-" * 30)
            for metric, value in results['overall'].items():
                report.append(f"{metric:20}: {value:.6f}")
        
        # Target-specific metrics
        for target, target_results in results.items():
            if target == 'overall':
                continue
                
            report.append(f"\n{target.upper()} PERFORMANCE:")
            report.append("-" * 30)
            
            # Accuracy metrics
            if 'accuracy' in target_results:
                report.append("Accuracy Metrics:")
                for metric, value in target_results['accuracy'].items():
                    report.append(f"  {metric:15}: {value:.6f}")
            
            # Uncertainty metrics
            if 'uncertainty' in target_results:
                report.append("Uncertainty Metrics:")
                for metric, value in target_results['uncertainty'].items():
                    if isinstance(value, dict):
                        report.append(f"  {metric:15}:")
                        for sub_metric, sub_value in value.items():
                            report.append(f"    {sub_metric:12}: {sub_value:.6f}")
                    else:
                        report.append(f"  {metric:15}: {value:.6f}")
            
            # Risk metrics
            if 'risk' in target_results:
                report.append("Risk Metrics:")
                for metric, value in target_results['risk'].items():
                    report.append(f"  {metric:15}: {value:.6f}")
        
        report.append("\n" + "=" * 60)
        
        return "\n".join(report)
    
    def save_results(self, results: Dict, path: str):
        """Save evaluation results to file."""
        import json
        
        # Convert numpy arrays to lists for JSON serialization
        def convert_numpy(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, dict):
                return {key: convert_numpy(value) for key, value in obj.items()}
            elif isinstance(obj, list):
                return [convert_numpy(item) for item in obj]
            else:
                return obj
        
        results_serializable = convert_numpy(results)
        
        with open(path, 'w') as f:
            json.dump(results_serializable, f, indent=2)
        
        logger.info(f"Evaluation results saved to {path}")
    
    def load_results(self, path: str) -> Dict:
        """Load evaluation results from file."""
        import json
        
        with open(path, 'r') as f:
            results = json.load(f)
        
        logger.info(f"Evaluation results loaded from {path}")
        return results
