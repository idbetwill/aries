"""
Transformer-based Probabilistic Forecaster

Implements Transformer neural networks for probabilistic forecasting
of energy market variables with attention mechanisms.
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Union, Any
import logging
import warnings
from pathlib import Path
import math

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader, TensorDataset
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    print("Warning: PyTorch not available. Install with: pip install torch")

warnings.filterwarnings('ignore')
logger = logging.getLogger(__name__)


class TransformerForecaster:
    """
    Transformer-based probabilistic forecaster for energy market variables.
    
    Implements Transformer neural networks with attention mechanisms for
    uncertainty quantification in energy price, demand, and supply forecasting.
    """
    
    def __init__(self, config: Dict = None):
        """
        Initialize the Transformer forecaster.
        
        Args:
            config: Configuration dictionary for the Transformer model
        """
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch is required for Transformer forecaster")
        
        self.config = config or self._default_config()
        self.model = None
        self.scaler_X = None
        self.scaler_y = None
        self.feature_names = []
        self.target_names = []
        self.is_trained = False
        
        # Initialize model architecture
        self._build_model()
        
    def _default_config(self) -> Dict:
        """Return default Transformer configuration."""
        return {
            'sequence_length': 168,  # hours
            'forecast_horizon': 24,  # hours
            'target_variables': ['price', 'demand', 'supply'],
            'features': ['price', 'demand', 'supply'],
            'd_model': 128,
            'nhead': 8,
            'num_layers': 4,
            'dropout': 0.1,
            'learning_rate': 0.0001,
            'epochs': 50,
            'batch_size': 16,
            'patience': 10,
            'quantiles': [0.05, 0.25, 0.5, 0.75, 0.95],
            'device': 'auto'  # 'auto', 'cpu', 'cuda'
        }
    
    def _build_model(self):
        """Build the Transformer model architecture."""
        if not TORCH_AVAILABLE:
            return
        
        # Set device
        if self.config['device'] == 'auto':
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(self.config['device'])
        
        logger.info(f"Using device: {self.device}")
        
        # Model will be built when we know input dimensions
        self.model = None
    
    def _create_transformer_model(self, input_size: int, output_size: int):
        """Create Transformer model with quantile regression."""
        
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch is required for Transformer forecasting. Install with: pip install torch")
        
        class PositionalEncoding(nn.Module):
            def __init__(self, d_model, max_len=5000):
                super(PositionalEncoding, self).__init__()
                
                pe = torch.zeros(max_len, d_model)
                position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
                div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                                   (-math.log(10000.0) / d_model))
                pe[:, 0::2] = torch.sin(position * div_term)
                pe[:, 1::2] = torch.cos(position * div_term)
                pe = pe.unsqueeze(0).transpose(0, 1)
                self.register_buffer('pe', pe)
            
            def forward(self, x):
                return x + self.pe[:x.size(0), :]
        
        class QuantileTransformer(nn.Module):
            def __init__(self, input_size, d_model, nhead, num_layers, dropout, 
                        output_size, quantiles, sequence_length):
                super(QuantileTransformer, self).__init__()
                
                self.d_model = d_model
                self.quantiles = quantiles
                self.num_quantiles = len(quantiles)
                self.sequence_length = sequence_length
                
                # Input projection
                self.input_projection = nn.Linear(input_size, d_model)
                
                # Positional encoding
                self.pos_encoding = PositionalEncoding(d_model, sequence_length)
                
                # Transformer encoder
                encoder_layer = nn.TransformerEncoderLayer(
                    d_model=d_model,
                    nhead=nhead,
                    dim_feedforward=d_model * 4,
                    dropout=dropout,
                    batch_first=True
                )
                self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)
                
                # Output layers for each quantile
                self.quantile_layers = nn.ModuleList([
                    nn.Sequential(
                        nn.Linear(d_model, d_model // 2),
                        nn.ReLU(),
                        nn.Dropout(dropout),
                        nn.Linear(d_model // 2, output_size)
                    ) for _ in range(self.num_quantiles)
                ])
                
                self.dropout = nn.Dropout(dropout)
                
            def forward(self, x):
                # Input projection
                x = self.input_projection(x)
                
                # Add positional encoding
                x = x.transpose(0, 1)  # (seq_len, batch, d_model)
                x = self.pos_encoding(x)
                x = x.transpose(0, 1)  # (batch, seq_len, d_model)
                
                # Transformer forward pass
                transformer_out = self.transformer(x)
                
                # Use the last output
                last_output = transformer_out[:, -1, :]
                last_output = self.dropout(last_output)
                
                # Generate predictions for each quantile
                quantile_predictions = []
                for layer in self.quantile_layers:
                    pred = layer(last_output)
                    quantile_predictions.append(pred)
                
                # Stack predictions
                predictions = torch.stack(quantile_predictions, dim=1)
                
                return predictions
        
        return QuantileTransformer(
            input_size=input_size,
            d_model=self.config['d_model'],
            nhead=self.config['nhead'],
            num_layers=self.config['num_layers'],
            dropout=self.config['dropout'],
            output_size=output_size,
            quantiles=self.config['quantiles'],
            sequence_length=self.config['sequence_length']
        )
    
    def _prepare_sequences(self, X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Prepare sequences for Transformer training.
        
        Args:
            X: Input features
            y: Target variables
            
        Returns:
            Tuple of (X_sequences, y_sequences)
        """
        sequence_length = self.config['sequence_length']
        forecast_horizon = self.config['forecast_horizon']
        
        X_sequences = []
        y_sequences = []
        
        for i in range(sequence_length, len(X) - forecast_horizon + 1):
            # Input sequence
            X_seq = X[i-sequence_length:i]
            X_sequences.append(X_seq)
            
            # Target sequence (forecast horizon)
            y_seq = y[i:i+forecast_horizon]
            y_sequences.append(y_seq)
        
        return np.array(X_sequences), np.array(y_sequences)
    
    def _create_data_loaders(self, X: np.ndarray, y: np.ndarray, 
                           X_val: np.ndarray = None, y_val: np.ndarray = None):
        """Create PyTorch data loaders."""
        # Convert to tensors
        X_tensor = torch.FloatTensor(X)
        y_tensor = torch.FloatTensor(y)
        
        # Create datasets
        train_dataset = TensorDataset(X_tensor, y_tensor)
        train_loader = DataLoader(train_dataset, 
                                batch_size=self.config['batch_size'], 
                                shuffle=True)
        
        val_loader = None
        if X_val is not None and y_val is not None:
            X_val_tensor = torch.FloatTensor(X_val)
            y_val_tensor = torch.FloatTensor(y_val)
            val_dataset = TensorDataset(X_val_tensor, y_val_tensor)
            val_loader = DataLoader(val_dataset, 
                                  batch_size=self.config['batch_size'], 
                                  shuffle=False)
        
        return train_loader, val_loader
    
    def train(self, X: np.ndarray, y: np.ndarray, 
              X_val: np.ndarray = None, y_val: np.ndarray = None) -> Dict:
        """
        Train the Transformer model.
        
        Args:
            X: Training input features
            y: Training target variables
            X_val: Validation input features
            y_val: Validation target variables
            
        Returns:
            Training results dictionary
        """
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch is required for Transformer training")
        
        logger.info("Starting Transformer training")
        
        # Store feature and target names
        self.feature_names = [f"feature_{i}" for i in range(X.shape[-1])]
        self.target_names = [f"target_{i}" for i in range(y.shape[-1])]
        
        # Prepare sequences
        X_seq, y_seq = self._prepare_sequences(X, y)
        
        if X_val is not None and y_val is not None:
            X_val_seq, y_val_seq = self._prepare_sequences(X_val, y_val)
        else:
            X_val_seq, y_val_seq = None, None
        
        # Create model
        input_size = X_seq.shape[-1]
        output_size = y_seq.shape[-1]
        self.model = self._create_transformer_model(input_size, output_size)
        self.model.to(self.device)
        
        # Create data loaders
        train_loader, val_loader = self._create_data_loaders(X_seq, y_seq, X_val_seq, y_val_seq)
        
        # Training setup
        criterion = nn.MSELoss()
        optimizer = optim.Adam(self.model.parameters(), lr=self.config['learning_rate'])
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)
        
        # Training loop
        train_losses = []
        val_losses = []
        best_val_loss = float('inf')
        patience_counter = 0
        
        for epoch in range(self.config['epochs']):
            # Training
            self.model.train()
            train_loss = 0.0
            
            for batch_X, batch_y in train_loader:
                batch_X = batch_X.to(self.device)
                batch_y = batch_y.to(self.device)
                
                optimizer.zero_grad()
                
                # Forward pass
                predictions = self.model(batch_X)
                
                # Calculate loss for each quantile
                total_loss = 0.0
                for i, quantile in enumerate(self.config['quantiles']):
                    pred_q = predictions[:, i, :]
                    loss = criterion(pred_q, batch_y)
                    total_loss += loss
                
                total_loss = total_loss / len(self.config['quantiles'])
                
                # Backward pass
                total_loss.backward()
                
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                
                optimizer.step()
                
                train_loss += total_loss.item()
            
            train_loss /= len(train_loader)
            train_losses.append(train_loss)
            
            # Validation
            val_loss = 0.0
            if val_loader:
                self.model.eval()
                with torch.no_grad():
                    for batch_X, batch_y in val_loader:
                        batch_X = batch_X.to(self.device)
                        batch_y = batch_y.to(self.device)
                        
                        predictions = self.model(batch_X)
                        
                        # Calculate validation loss
                        total_loss = 0.0
                        for i, quantile in enumerate(self.config['quantiles']):
                            pred_q = predictions[:, i, :]
                            loss = criterion(pred_q, batch_y)
                            total_loss += loss
                        
                        total_loss = total_loss / len(self.config['quantiles'])
                        val_loss += total_loss.item()
                
                val_loss /= len(val_loader)
                val_losses.append(val_loss)
                
                # Learning rate scheduling
                scheduler.step(val_loss)
                
                # Early stopping
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    patience_counter = 0
                    # Save best model
                    self.best_model_state = self.model.state_dict().copy()
                else:
                    patience_counter += 1
                
                if patience_counter >= self.config['patience']:
                    logger.info(f"Early stopping at epoch {epoch}")
                    break
            
            if epoch % 10 == 0:
                logger.info(f"Epoch {epoch}: Train Loss = {train_loss:.6f}, Val Loss = {val_loss:.6f}")
        
        # Load best model
        if hasattr(self, 'best_model_state'):
            self.model.load_state_dict(self.best_model_state)
        
        self.is_trained = True
        
        results = {
            'train_losses': train_losses,
            'val_losses': val_losses,
            'best_val_loss': best_val_loss,
            'epochs_trained': len(train_losses)
        }
        
        logger.info("Transformer training completed")
        return results
    
    def predict(self, X: np.ndarray, horizon: int = None) -> Dict:
        """
        Make predictions using the trained Transformer model.
        
        Args:
            X: Input features
            horizon: Forecast horizon (if None, uses config default)
            
        Returns:
            Dictionary with predictions and uncertainty
        """
        if not self.is_trained:
            raise RuntimeError("Model must be trained before making predictions")
        
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch is required for Transformer predictions")
        
        horizon = horizon or self.config['forecast_horizon']
        
        self.model.eval()
        predictions = {}
        
        with torch.no_grad():
            # Prepare input sequence
            sequence_length = self.config['sequence_length']
            if len(X) < sequence_length:
                # Pad with zeros if sequence is too short
                X_padded = np.zeros((sequence_length, X.shape[1]))
                X_padded[-len(X):] = X
                X = X_padded
            
            # Take the last sequence
            X_seq = X[-sequence_length:].reshape(1, sequence_length, -1)
            X_tensor = torch.FloatTensor(X_seq).to(self.device)
            
            # Make prediction
            pred = self.model(X_tensor)
            pred = pred.cpu().numpy()
            
            # Extract predictions for each target variable
            for i, target in enumerate(self.config['target_variables']):
                if i < pred.shape[-1]:
                    target_predictions = pred[0, :, i]  # Shape: (num_quantiles, horizon)
                    
                    # Organize by quantile
                    quantile_predictions = {}
                    for j, quantile in enumerate(self.config['quantiles']):
                        if j < len(target_predictions):
                            quantile_predictions[quantile] = target_predictions[j]
                    
                    predictions[target] = {
                        'quantiles': quantile_predictions,
                        'mean': np.mean(target_predictions, axis=0),
                        'std': np.std(target_predictions, axis=0)
                    }
        
        return predictions
    
    def save(self, path: str):
        """Save the trained model."""
        if not self.is_trained:
            raise RuntimeError("Model must be trained before saving")
        
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        # Save model state
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'config': self.config,
            'feature_names': self.feature_names,
            'target_names': self.target_names,
            'is_trained': self.is_trained
        }, path)
        
        logger.info(f"Transformer model saved to {path}")
    
    def load(self, path: str):
        """Load a trained model."""
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch is required for loading Transformer models")
        
        path = Path(path)
        
        if not path.exists():
            raise FileNotFoundError(f"Model file not found: {path}")
        
        # Load model state
        checkpoint = torch.load(path, map_location=self.device)
        
        self.config = checkpoint['config']
        self.feature_names = checkpoint['feature_names']
        self.target_names = checkpoint['target_names']
        self.is_trained = checkpoint['is_trained']
        
        # Recreate model
        input_size = len(self.feature_names)
        output_size = len(self.target_names)
        self.model = self._create_transformer_model(input_size, output_size)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.to(self.device)
        
        logger.info(f"Transformer model loaded from {path}")
    
    def get_model_info(self) -> Dict:
        """Get information about the model."""
        if not self.is_trained:
            return {'is_trained': False}
        
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        
        return {
            'is_trained': self.is_trained,
            'config': self.config,
            'total_parameters': total_params,
            'trainable_parameters': trainable_params,
            'feature_names': self.feature_names,
            'target_names': self.target_names,
            'device': str(self.device)
        }
