# Aries Trading Agent - Usage Guide

## 🚀 Quick Start

### 1. Installation

```bash
# Clone the repository
git clone https://github.com/your-repo/aries.git
cd aries

# Install dependencies
pip install -r requirements.txt

# Install in development mode
pip install -e .
```

### 2. Web Interface (Recommended)

```bash
# Launch the Streamlit web interface
streamlit run app.py
```

This will open a web interface at `http://localhost:8501` with:
- 📈 Real-time dashboard
- 🤖 Agent configuration and training
- 📊 Performance monitoring
- 🔮 Forecasting capabilities
- ⚙️ Settings and controls

### 3. Command Line Interface

```bash
# Train the agent
python -m aries train --algorithm PPO --timesteps 100000

# Run backtesting
python -m aries backtest --model-path ./models --test-data-days 30

# Generate forecasts
python -m aries forecast --data-days 7 --horizon 24

# Collect market data
python -m aries collect-data --data-days 30 --output-file data.csv
```

## 📊 Data Sources

### XM API Integration
- **Automatic**: No API key required
- **Data**: Historical energy prices, demand, supply
- **Frequency**: Hourly updates
- **Coverage**: Colombian wholesale market

### San Andrés & Providencia
- **Specialized**: Island-specific energy data
- **Features**: Local consumption, generation, storage
- **Focus**: Renewable energy integration
- **Autonomy**: Island energy independence metrics

### Synthetic Data
- **Development**: For testing and demonstration
- **Realistic**: Based on Colombian market patterns
- **Configurable**: Adjustable parameters

## 🤖 Agent Configuration

### Risk Management
```yaml
risk:
  risk_aversion_lambda: 0.5      # Risk aversion coefficient
  max_position_size: 1000.0      # Maximum position size (MWh)
  max_drawdown: 0.15             # Maximum drawdown (15%)
  var_confidence: 0.05           # Value at Risk confidence
  cvar_confidence: 0.05          # Conditional VaR confidence
```

### Training Parameters
```yaml
training:
  total_timesteps: 100000        # Total training steps
  learning_rate: 0.0003          # Learning rate
  batch_size: 64                 # Batch size
  gamma: 0.99                    # Discount factor
  ent_coef: 0.01                 # Entropy coefficient
```

### Forecasting
```yaml
forecasting:
  forecast_horizon: 24            # Hours ahead
  models:
    lstm: true                   # LSTM model
    transformer: true             # Transformer model
    ensemble: true                # Ensemble method
  uncertainty_quantification: "quantile_regression"
```

## 🔮 Forecasting Capabilities

### Probabilistic Predictions
- **Price Forecasting**: Energy price predictions with uncertainty
- **Demand Forecasting**: Consumption predictions
- **Supply Forecasting**: Generation predictions
- **Uncertainty Quantification**: Confidence intervals and risk metrics

### Model Ensemble
- **LSTM**: Long Short-Term Memory networks
- **Transformer**: Attention-based models
- **Ensemble**: Weighted combination of models
- **Uncertainty**: Quantile regression and Monte Carlo methods

## 🛡️ Risk Management

### Risk Metrics
- **VaR**: Value at Risk (5% confidence)
- **CVaR**: Conditional Value at Risk
- **Volatility**: Price volatility tracking
- **Sharpe Ratio**: Risk-adjusted returns
- **Maximum Drawdown**: Worst-case scenario

### Dynamic Risk Adjustment
- **Market Regime Detection**: Automatic regime identification
- **Volatility-Based**: Position sizing based on market volatility
- **Correlation Monitoring**: Multi-asset correlation tracking
- **Real-time Adjustment**: Continuous risk parameter updates

## 📈 Performance Monitoring

### Key Metrics
- **Total Return**: Overall portfolio performance
- **Sharpe Ratio**: Risk-adjusted returns
- **Maximum Drawdown**: Largest peak-to-trough decline
- **Success Rate**: Percentage of profitable trades
- **Volatility**: Portfolio volatility

### Real-time Monitoring
- **Portfolio Value**: Current portfolio value
- **Position Size**: Current position in MWh
- **Risk Metrics**: Real-time risk calculations
- **Trade History**: Recent trading activity

## 🏝️ San Andrés Specialization

### Island-Specific Features
- **Autonomy Hours**: Energy independence duration
- **Renewable Percentage**: Clean energy contribution
- **Storage Management**: Battery storage optimization
- **Diesel Backup**: Backup generation management

### Local Market Dynamics
- **Weather Dependency**: Solar and wind generation
- **Tourism Impact**: Seasonal consumption patterns
- **Grid Stability**: Island grid management
- **Cost Optimization**: Minimize diesel usage

## 🔧 Advanced Configuration

### Environment Variables
```bash
export ARIES_RISK_AVERSION=0.5
export ARIES_INITIAL_CAPITAL=100000
export ARIES_ALGORITHM=PPO
export ARIES_DATA_SOURCE=XM
```

### Configuration File
```yaml
# config.yaml
agent:
  algorithm: "PPO"
  risk_aversion_lambda: 0.5
  initial_capital: 100000.0

risk:
  max_drawdown: 0.15
  var_confidence: 0.05

training:
  total_timesteps: 100000
  learning_rate: 0.0003
```

## 📊 Backtesting

### Historical Testing
```python
# Run backtest
results = agent.backtest(test_data, initial_capital=100000)

# Analyze results
performance = results['performance']
print(f"Total Return: {performance['total_return']:.2%}")
print(f"Sharpe Ratio: {performance['sharpe_ratio']:.2f}")
print(f"Max Drawdown: {performance['max_drawdown']:.2%}")
```

### Performance Analysis
- **Return Analysis**: Historical return patterns
- **Risk Analysis**: Risk metric evolution
- **Trade Analysis**: Individual trade performance
- **Benchmark Comparison**: Against market indices

## 🚀 Deployment

### Paper Trading
```python
# Enable paper trading mode
agent.config['deployment']['trading']['paper_trading'] = True

# Run live simulation
agent.deploy()
```

### Production Deployment
```python
# Configure for production
agent.config['deployment']['mode'] = 'production'
agent.config['deployment']['trading']['paper_trading'] = False

# Deploy with monitoring
agent.deploy(monitoring=True)
```

## 🔍 Troubleshooting

### Common Issues

#### 1. Import Errors
```bash
# Install missing dependencies
pip install -r requirements.txt

# Install PyTorch separately if needed
pip install torch torchvision torchaudio
```

#### 2. Data Collection Issues
```python
# Check API connectivity
from aries.data import XMDataCollector
collector = XMDataCollector()
data = collector.get_historical_prices(start_date, end_date)
```

#### 3. Training Issues
```python
# Reduce training complexity for testing
agent.config['training']['total_timesteps'] = 10000
agent.config['training']['batch_size'] = 32
```

#### 4. Memory Issues
```python
# Reduce batch size and sequence length
agent.config['training']['batch_size'] = 16
agent.config['forecasting']['sequence_length'] = 72
```

### Logging
```python
import logging
logging.basicConfig(level=logging.INFO)

# Enable debug logging
logging.getLogger('aries').setLevel(logging.DEBUG)
```

## 📚 Examples

### Basic Usage
```python
from aries import RiskAverseTradingAgent, EnergyMarketEnv, ProbabilisticForecaster

# Create components
env = EnergyMarketEnv(data=your_data)
forecaster = ProbabilisticForecaster()
agent = RiskAverseTradingAgent(env, forecaster)

# Train agent
agent.train(episodes=10000)

# Run backtest
results = agent.backtest(test_data)
```

### Advanced Configuration
```python
# Custom configuration
config = {
    'risk': {
        'risk_aversion_lambda': 0.7,
        'max_drawdown': 0.1
    },
    'training': {
        'learning_rate': 0.0001,
        'batch_size': 128
    }
}

agent = RiskAverseTradingAgent(env, forecaster, config=config)
```

### San Andrés Focus
```python
from aries.data import SanAndresDataCollector

# Collect San Andrés data
sa_collector = SanAndresDataCollector()
sa_data = sa_collector.get_historical_data(start_date, end_date)

# Use island-specific data
env = EnergyMarketEnv(data=sa_data)
agent = RiskAverseTradingAgent(env, forecaster)
```

## 🤝 Contributing

### Development Setup
```bash
# Install development dependencies
pip install -e ".[dev]"

# Run tests
pytest

# Format code
black aries/

# Type checking
mypy aries/
```

### Code Style
- Follow PEP 8
- Use type hints
- Document functions
- Write tests

## 📞 Support

- **Documentation**: [aries.readthedocs.io](https://aries.readthedocs.io)
- **Issues**: [GitHub Issues](https://github.com/your-repo/aries/issues)
- **Discussions**: [GitHub Discussions](https://github.com/your-repo/aries/discussions)
- **Email**: aries@example.com

## 📄 License

MIT License - see [LICENSE](LICENSE) file for details.
