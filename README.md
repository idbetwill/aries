# Aries: Risk-Averse Trading Agent for Colombian Energy Market

## Overview
Aries is a sophisticated Reinforcement Learning-based trading agent designed for the Colombian wholesale energy market (XM). The system combines probabilistic forecasting with risk-averse decision-making to optimize energy trading strategies.

## Features
- **Probabilistic Forecasting**: LSTM/Transformer-based models for price prediction with uncertainty quantification
- **Risk-Averse RL Agent**: PPO/SAC algorithms with CVaR-based risk management
- **Market Simulation**: Realistic trading environment using historical data
- **Real-time Integration**: XM API integration for live market data
- **San Andrés Focus**: Specialized for San Andrés and Providencia energy market dynamics
- **Web Interface**: Streamlit-based dashboard for monitoring and control

## Architecture

### Phase 1: Data Foundation
- Historical energy price data from XM API
- Weather and external factors
- San Andrés specific data integration

### Phase 2: Probabilistic Forecasting
- LSTM/Transformer models for price prediction
- Quantile regression for uncertainty bounds
- Mixture Density Networks for distribution modeling

### Phase 3: Risk-Averse RL Agent
- State representation with forecast distributions
- Action space: buy/sell/hold decisions
- Reward function: Profit - λ × CVaR

### Phase 4: Integration & Deployment
- Real-time market simulation
- Backtesting framework
- Performance monitoring dashboard

## Installation

```bash
# Clone the repository
git clone https://github.com/your-repo/aries.git
cd aries

# Install dependencies
pip install -r requirements.txt

# Set up environment variables
cp .env.example .env
# Edit .env with your configuration
```

## Quick Start

```python
from aries.agent import RiskAverseTradingAgent
from aries.environment import EnergyMarketEnv
from aries.forecaster import ProbabilisticForecaster

# Initialize components
forecaster = ProbabilisticForecaster()
env = EnergyMarketEnv()
agent = RiskAverseTradingAgent(env, forecaster)

# Train the agent
agent.train(episodes=10000)

# Run backtesting
results = agent.backtest(test_data)
```

## Usage

### Web Interface
```bash
streamlit run app.py
```

### Command Line
```bash
python -m aries.cli train --episodes 10000 --risk-aversion 0.5
python -m aries.cli backtest --model-path models/best_model.zip
```

## Configuration

The system can be configured through environment variables or config files:

- `RISK_AVERSION_LAMBDA`: Risk aversion coefficient (default: 0.5)
- `FORECAST_HORIZON`: Prediction horizon in hours (default: 24)
- `TRADING_FREQUENCY`: Trading frequency in minutes (default: 60)
- `MAX_POSITION_SIZE`: Maximum position size (default: 1000)

## Performance Metrics

- **Financial**: Total return, Sharpe ratio, Maximum drawdown
- **Risk**: CVaR, Value at Risk (VaR)
- **Forecasting**: CRPS, MAE, RMSE
- **Trading**: Win rate, Average trade duration

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Submit a pull request

## License

MIT License - see LICENSE file for details
