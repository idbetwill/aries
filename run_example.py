"""
Example script demonstrating the Aries Trading Agent

This script shows how to use the risk-averse trading agent
for energy market trading with San Andrés focus.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    """Main example function."""
    print("⚡ Aries Trading Agent - Example Usage")
    print("=" * 50)
    
    try:
        # Import Aries components
        from aries import RiskAverseTradingAgent, EnergyMarketEnv, ProbabilisticForecaster, DataManager
        from aries.data import XMDataCollector, SanAndresDataCollector
        
        print("✅ Aries components imported successfully")
        
        # 1. Data Collection
        print("\n📊 Step 1: Data Collection")
        print("-" * 30)
        
        # Create data manager
        data_manager = DataManager()
        
        # Collect data for the last 30 days
        end_date = datetime.now()
        start_date = end_date - timedelta(days=30)
        
        print(f"Collecting data from {start_date.date()} to {end_date.date()}")
        
        # Try to collect real data, fallback to synthetic
        try:
            data = data_manager.collect_market_data(start_date, end_date)
            if data.empty:
                raise ValueError("No real data available")
            print(f"✅ Collected {len(data)} records from APIs")
        except Exception as e:
            print(f"⚠️ API data not available: {e}")
            print("🔄 Generating synthetic data for demonstration...")
            data = generate_synthetic_data(days=30)
            print(f"✅ Generated {len(data)} synthetic records")
        
        # 2. Environment Setup
        print("\n🌍 Step 2: Environment Setup")
        print("-" * 30)
        
        # Create trading environment
        environment = EnergyMarketEnv(
            data=data,
            initial_capital=100000.0,
            risk_aversion_lambda=0.5
        )
        
        print("✅ Trading environment created")
        print(f"   - Initial capital: COP 100,000")
        print(f"   - Risk aversion: 0.5")
        print(f"   - Max steps: {environment.max_steps}")
        
        # 3. Forecasting Setup
        print("\n🔮 Step 3: Forecasting Setup")
        print("-" * 30)
        
        # Create probabilistic forecaster
        forecaster = ProbabilisticForecaster()
        
        print("✅ Probabilistic forecaster created")
        print("   - LSTM model: Enabled")
        print("   - Transformer model: Enabled")
        print("   - Ensemble method: Weighted average")
        
        # 4. Agent Creation
        print("\n🤖 Step 4: Agent Creation")
        print("-" * 30)
        
        # Create risk-averse trading agent
        agent = RiskAverseTradingAgent(
            environment=environment,
            forecaster=forecaster,
            algorithm='PPO'
        )
        
        print("✅ Risk-averse trading agent created")
        print(f"   - Algorithm: PPO")
        print(f"   - Risk management: Enabled")
        print(f"   - Forecasting integration: Enabled")
        
        # 5. Training
        print("\n🏋️ Step 5: Agent Training")
        print("-" * 30)
        
        print("Training agent for 10,000 timesteps...")
        training_results = agent.train(episodes=1000)  # Reduced for demo
        
        print("✅ Training completed")
        print(f"   - Training time: {training_results.get('training_time', 'N/A')}")
        print(f"   - Success: {training_results.get('success', False)}")
        
        # 6. Backtesting
        print("\n📊 Step 6: Backtesting")
        print("-" * 30)
        
        # Generate test data
        test_data = generate_synthetic_data(days=7)
        
        print("Running backtest on 7 days of test data...")
        backtest_results = agent.backtest(test_data, verbose=False)
        
        # Display results
        performance = backtest_results['performance']
        print("✅ Backtest completed")
        print(f"   - Total return: {performance['total_return']:.2%}")
        print(f"   - Sharpe ratio: {performance['sharpe_ratio']:.2f}")
        print(f"   - Max drawdown: {performance['max_drawdown']:.2%}")
        print(f"   - Success rate: {performance['success_rate']:.2%}")
        print(f"   - Total trades: {performance['total_trades']}")
        
        # 7. Forecasting Demo
        print("\n🔮 Step 7: Forecasting Demo")
        print("-" * 30)
        
        # Train forecaster
        print("Training forecaster...")
        forecaster.train(data)
        
        # Generate forecast
        print("Generating 24-hour forecast...")
        forecast = forecaster.predict(data.tail(168), horizon=24)  # Last week of data
        
        if forecast:
            print("✅ Forecast generated successfully")
            for target, pred in forecast.items():
                if 'prediction' in pred:
                    mean_pred = pred['prediction']
                    print(f"   - {target.title()}: {mean_pred[0]:.2f} → {mean_pred[-1]:.2f}")
        else:
            print("⚠️ Forecast not available (forecaster not trained)")
        
        # 8. San Andrés Specific Features
        print("\n🏝️ Step 8: San Andrés Features")
        print("-" * 30)
        
        # Create San Andrés data collector
        sa_collector = SanAndresDataCollector()
        
        # Get San Andrés specific data
        sa_data = sa_collector.get_historical_data(start_date, end_date)
        
        if not sa_data.empty:
            print("✅ San Andrés data collected")
            print(f"   - Records: {len(sa_data)}")
            print(f"   - Columns: {list(sa_data.columns)}")
            
            # Show island-specific metrics
            if 'island_autonomy_hours' in sa_data.columns:
                avg_autonomy = sa_data['island_autonomy_hours'].mean()
                print(f"   - Average autonomy: {avg_autonomy:.1f} hours")
            
            if 'renewable_percentage' in sa_data.columns:
                avg_renewable = sa_data['renewable_percentage'].mean()
                print(f"   - Average renewable: {avg_renewable:.1f}%")
        else:
            print("⚠️ San Andrés data not available (using synthetic)")
        
        # 9. Risk Management Demo
        print("\n🛡️ Step 9: Risk Management")
        print("-" * 30)
        
        # Get agent risk info
        agent_info = agent.get_agent_info()
        risk_info = agent_info.get('risk_manager_info', {})
        
        if risk_info:
            print("✅ Risk management active")
            print(f"   - Risk aversion: {risk_info['config']['risk_aversion_lambda']}")
            print(f"   - Max position: {risk_info['config']['max_position_size']}")
            print(f"   - Max drawdown: {risk_info['config']['max_drawdown']:.1%}")
        
        # 10. Summary
        print("\n📋 Summary")
        print("=" * 50)
        print("✅ Aries Trading Agent demonstration completed successfully!")
        print("\nKey Features Demonstrated:")
        print("  🔹 Risk-averse reinforcement learning")
        print("  🔹 Probabilistic forecasting with uncertainty")
        print("  🔹 San Andrés island-specific trading")
        print("  🔹 Comprehensive risk management")
        print("  🔹 Backtesting and performance evaluation")
        print("  🔹 XM API integration")
        print("\nNext Steps:")
        print("  🚀 Run 'streamlit run app.py' for web interface")
        print("  🖥️ Run 'python -m aries train' for CLI training")
        print("  📊 Run 'python -m aries backtest' for backtesting")
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("Please install Aries dependencies: pip install -r requirements.txt")
    except Exception as e:
        print(f"❌ Error: {e}")
        logger.error(f"Example execution failed: {e}", exc_info=True)

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

if __name__ == "__main__":
    main()
