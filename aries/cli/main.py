"""
Main CLI interface for Aries Trading Agent

Provides command-line interface for training, backtesting,
and managing the risk-averse trading agent.
"""

import click
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
import logging
import json

from .. import RiskAverseTradingAgent, EnergyMarketEnv, ProbabilisticForecaster, DataManager

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@click.group()
def cli():
    """Aries Trading Agent - Risk-Averse Energy Trading with RL"""
    pass

@cli.command()
@click.option('--algorithm', default='PPO', type=click.Choice(['PPO', 'SAC']), help='RL algorithm to use')
@click.option('--risk-aversion', default=0.5, type=float, help='Risk aversion coefficient')
@click.option('--initial-capital', default=100000, type=int, help='Initial capital in COP')
@click.option('--timesteps', default=100000, type=int, help='Total training timesteps')
@click.option('--data-days', default=30, type=int, help='Days of historical data to use')
@click.option('--output-dir', default='./models', type=click.Path(), help='Output directory for models')
@click.option('--config', type=click.Path(exists=True), help='Configuration file path')
def train(algorithm, risk_aversion, initial_capital, timesteps, data_days, output_dir, config):
    """Train the risk-averse trading agent"""
    
    click.echo(f"🚀 Starting Aries training with {algorithm} algorithm")
    
    try:
        # Load configuration if provided
        agent_config = {}
        if config:
            with open(config, 'r') as f:
                agent_config = json.load(f)
        
        # Create data manager
        data_manager = DataManager()
        
        # Get training data
        click.echo("📊 Collecting market data...")
        end_date = datetime.now()
        start_date = end_date - timedelta(days=data_days)
        data = data_manager.collect_market_data(start_date, end_date)
        
        if data.empty:
            click.echo("⚠️ No data available, generating synthetic data...")
            data = generate_synthetic_data(days=data_days)
        
        # Create environment
        click.echo("🌍 Creating trading environment...")
        environment = EnergyMarketEnv(
            data=data,
            initial_capital=initial_capital,
            risk_aversion_lambda=risk_aversion
        )
        
        # Create forecaster
        click.echo("🔮 Initializing probabilistic forecaster...")
        forecaster = ProbabilisticForecaster()
        
        # Create agent
        click.echo("🤖 Creating risk-averse trading agent...")
        agent = RiskAverseTradingAgent(
            environment=environment,
            forecaster=forecaster,
            algorithm=algorithm,
            config=agent_config
        )
        
        # Train the agent
        click.echo(f"🏋️ Training agent for {timesteps} timesteps...")
        results = agent.train(episodes=None)
        
        # Save the agent
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        click.echo(f"💾 Saving agent to {output_path}")
        agent.save_agent(str(output_path))
        
        click.echo("✅ Training completed successfully!")
        click.echo(f"📊 Training results: {json.dumps(results, indent=2, default=str)}")
        
    except Exception as e:
        click.echo(f"❌ Error during training: {str(e)}")
        raise click.Abort()

@cli.command()
@click.option('--model-path', required=True, type=click.Path(exists=True), help='Path to trained model')
@click.option('--test-data-days', default=30, type=int, help='Days of test data to use')
@click.option('--initial-capital', default=100000, type=int, help='Initial capital for backtesting')
@click.option('--output-file', type=click.Path(), help='Output file for backtest results')
def backtest(model_path, test_data_days, initial_capital, output_file):
    """Run backtesting on the trained agent"""
    
    click.echo(f"📊 Starting backtest with {test_data_days} days of test data")
    
    try:
        # Load the agent
        click.echo("🤖 Loading trained agent...")
        agent = RiskAverseTradingAgent.load_agent(model_path)
        
        # Get test data
        click.echo("📊 Collecting test data...")
        data_manager = DataManager()
        end_date = datetime.now()
        start_date = end_date - timedelta(days=test_data_days)
        test_data = data_manager.collect_market_data(start_date, end_date)
        
        if test_data.empty:
            click.echo("⚠️ No test data available, generating synthetic data...")
            test_data = generate_synthetic_data(days=test_data_days)
        
        # Run backtest
        click.echo("🔄 Running backtest...")
        results = agent.backtest(test_data, initial_capital=initial_capital, verbose=False)
        
        # Display results
        performance = results['performance']
        click.echo("\n📈 Backtest Results:")
        click.echo(f"  Total Return: {performance['total_return']:.2%}")
        click.echo(f"  Sharpe Ratio: {performance['sharpe_ratio']:.2f}")
        click.echo(f"  Max Drawdown: {performance['max_drawdown']:.2%}")
        click.echo(f"  Volatility: {performance['volatility']:.2%}")
        click.echo(f"  Success Rate: {performance['success_rate']:.2%}")
        click.echo(f"  Total Trades: {performance['total_trades']}")
        
        # Save results if output file specified
        if output_file:
            click.echo(f"💾 Saving results to {output_file}")
            with open(output_file, 'w') as f:
                # Convert numpy arrays to lists for JSON serialization
                serializable_results = convert_for_json(results)
                json.dump(serializable_results, f, indent=2, default=str)
        
        click.echo("✅ Backtest completed successfully!")
        
    except Exception as e:
        click.echo(f"❌ Error during backtest: {str(e)}")
        raise click.Abort()

@cli.command()
@click.option('--data-days', default=7, type=int, help='Days of data to use for forecasting')
@click.option('--horizon', default=24, type=int, help='Forecast horizon in hours')
@click.option('--output-file', type=click.Path(), help='Output file for forecast results')
def forecast(data_days, horizon, output_file):
    """Generate probabilistic forecasts"""
    
    click.echo(f"🔮 Generating {horizon}-hour forecast using {data_days} days of data")
    
    try:
        # Create forecaster
        forecaster = ProbabilisticForecaster()
        
        # Get data
        data_manager = DataManager()
        end_date = datetime.now()
        start_date = end_date - timedelta(days=data_days)
        data = data_manager.collect_market_data(start_date, end_date)
        
        if data.empty:
            click.echo("⚠️ No data available, generating synthetic data...")
            data = generate_synthetic_data(days=data_days)
        
        # Train forecaster (simplified)
        click.echo("🏋️ Training forecaster...")
        forecaster.train(data)
        
        # Generate forecast
        click.echo("🔮 Generating forecast...")
        forecast_results = forecaster.predict(data, horizon=horizon)
        
        # Display results
        click.echo("\n📊 Forecast Results:")
        for target, pred in forecast_results.items():
            if 'prediction' in pred:
                mean_pred = pred['prediction']
                click.echo(f"  {target.title()}: {mean_pred[0]:.2f} (1h), {mean_pred[-1]:.2f} ({horizon}h)")
        
        # Save results if output file specified
        if output_file:
            click.echo(f"💾 Saving forecast to {output_file}")
            with open(output_file, 'w') as f:
                serializable_forecast = convert_for_json(forecast_results)
                json.dump(serializable_forecast, f, indent=2, default=str)
        
        click.echo("✅ Forecast completed successfully!")
        
    except Exception as e:
        click.echo(f"❌ Error during forecasting: {str(e)}")
        raise click.Abort()

@cli.command()
@click.option('--data-days', default=30, type=int, help='Days of data to collect')
@click.option('--output-file', type=click.Path(), help='Output file for data')
def collect_data(data_days, output_file):
    """Collect market data from XM API and San Andrés"""
    
    click.echo(f"📊 Collecting {data_days} days of market data")
    
    try:
        # Create data manager
        data_manager = DataManager()
        
        # Collect data
        end_date = datetime.now()
        start_date = end_date - timedelta(days=data_days)
        data = data_manager.collect_market_data(start_date, end_date)
        
        if data.empty:
            click.echo("⚠️ No data collected from APIs, generating synthetic data...")
            data = generate_synthetic_data(days=data_days)
        
        # Display summary
        click.echo(f"📈 Collected {len(data)} records")
        click.echo(f"📅 Date range: {data['timestamp'].min()} to {data['timestamp'].max()}")
        click.echo(f"💰 Price range: {data['price'].min():.2f} - {data['price'].max():.2f} COP/MWh")
        
        # Save data if output file specified
        if output_file:
            click.echo(f"💾 Saving data to {output_file}")
            data.to_csv(output_file, index=False)
        
        click.echo("✅ Data collection completed successfully!")
        
    except Exception as e:
        click.echo(f"❌ Error during data collection: {str(e)}")
        raise click.Abort()

@cli.command()
@click.option('--model-path', required=True, type=click.Path(exists=True), help='Path to trained model')
@click.option('--config', type=click.Path(exists=True), help='Configuration file path')
def deploy(model_path, config):
    """Deploy the trained agent for live trading"""
    
    click.echo("🚀 Deploying Aries trading agent")
    
    try:
        # Load the agent
        click.echo("🤖 Loading trained agent...")
        agent = RiskAverseTradingAgent.load_agent(model_path)
        
        # Load configuration if provided
        deploy_config = {}
        if config:
            with open(config, 'r') as f:
                deploy_config = json.load(f)
        
        # This would implement actual deployment logic
        # For now, just show agent info
        agent_info = agent.get_agent_info()
        
        click.echo("📊 Agent Information:")
        click.echo(f"  Algorithm: {agent_info['algorithm']}")
        click.echo(f"  Trained: {agent_info['is_trained']}")
        click.echo(f"  Training Sessions: {agent_info['training_history']}")
        
        click.echo("⚠️ Live deployment not implemented yet - this is a placeholder")
        click.echo("✅ Deployment preparation completed!")
        
    except Exception as e:
        click.echo(f"❌ Error during deployment: {str(e)}")
        raise click.Abort()

def generate_synthetic_data(days=30):
    """Generate synthetic market data for demonstration."""
    # Generate hourly data
    timestamps = pd.date_range(start=datetime.now() - timedelta(days=days), 
                             end=datetime.now(), freq='H')
    
    # Generate price data with realistic patterns
    base_price = 200.0
    price_trend = np.linspace(0, 0.1, len(timestamps))
    price_cycle = 10 * np.sin(2 * np.pi * np.arange(len(timestamps)) / 24)
    price_noise = np.random.normal(0, 5, len(timestamps))
    
    prices = base_price + price_trend + price_cycle + price_noise
    prices = np.maximum(prices, 50.0)
    
    # Generate demand and supply
    base_demand = 1000.0
    demand_cycle = 200 * np.sin(2 * np.pi * np.arange(len(timestamps)) / 24)
    demand_noise = np.random.normal(0, 50, len(timestamps))
    demand = base_demand + demand_cycle + demand_noise
    demand = np.maximum(demand, 100.0)
    
    base_supply = 1100.0
    supply_cycle = 150 * np.sin(2 * np.pi * np.arange(len(timestamps)) / 24 + np.pi/4)
    supply_noise = np.random.normal(0, 80, len(timestamps))
    supply = base_supply + supply_cycle + supply_noise
    supply = np.maximum(supply, 200.0)
    
    # Create DataFrame
    data = pd.DataFrame({
        'timestamp': timestamps,
        'price': prices,
        'price_change': np.gradient(prices),
        'price_volatility_24h': pd.Series(prices).rolling(24).std().fillna(0),
        'demand': demand,
        'supply': supply,
        'demand_supply_ratio': demand / supply,
        'hour': timestamps.hour,
        'day_of_week': timestamps.dayofweek,
        'is_weekend': (timestamps.dayofweek >= 5).astype(int)
    })
    
    return data

def convert_for_json(obj):
    """Convert numpy arrays and other non-serializable objects for JSON."""
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, dict):
        return {key: convert_for_json(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_for_json(item) for item in obj]
    else:
        return obj

if __name__ == '__main__':
    cli()
