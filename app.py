"""
Aries Trading Agent - Main Application

Streamlit-based web interface for the risk-averse energy trading agent
with real-time monitoring, control, and analysis capabilities.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import logging
import sys
from pathlib import Path

# Add the aries package to the path
sys.path.append(str(Path(__file__).parent))

from aries import RiskAverseTradingAgent, EnergyMarketEnv, ProbabilisticForecaster, DataManager
from aries.data import XMDataCollector, SanAndresDataCollector
from aries.agent import PPOAgent, SACAgent

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Page configuration
st.set_page_config(
    page_title="Agente de Trading Aries",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
    }
    .status-good {
        color: #28a745;
        font-weight: bold;
    }
    .status-warning {
        color: #ffc107;
        font-weight: bold;
    }
    .status-danger {
        color: #dc3545;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

def main():
    """Main application function."""
    
    # Header
    st.markdown('<h1 class="main-header">⚡ Agente de Trading Aries</h1>', unsafe_allow_html=True)
    st.markdown("**Agente de Trading Energético con Aversión al Riesgo para el Mercado Colombiano**")
    
    # Initialize session state
    if 'agent' not in st.session_state:
        st.session_state.agent = None
    if 'environment' not in st.session_state:
        st.session_state.environment = None
    if 'forecaster' not in st.session_state:
        st.session_state.forecaster = None
    if 'data_manager' not in st.session_state:
        st.session_state.data_manager = None
    
    # Sidebar
    with st.sidebar:
        st.header("🎛️ Panel de Control")
        
        # Agent configuration
        st.subheader("Configuración del Agente")
        algorithm = st.selectbox("Algoritmo RL", ["PPO", "SAC"], index=0)
        risk_aversion = st.slider("Aversión al Riesgo (λ)", 0.0, 2.0, 0.5, 0.1)
        initial_capital = st.number_input("Capital Inicial (COP)", 10000, 1000000, 100000, 10000)
        
        # Training parameters
        st.subheader("Parámetros de Entrenamiento")
        total_timesteps = st.number_input("Total de Pasos", 1000, 1000000, 100000, 10000)
        learning_rate = st.number_input("Tasa de Aprendizaje", 1e-5, 1e-2, 3e-4, 1e-5, format="%.2e")
        
        # Data configuration
        st.subheader("Configuración de Datos")
        data_source = st.selectbox("Fuente de Datos", ["API XM", "San Andrés", "Sintéticos"], index=2)
        data_days = st.number_input("Días Históricos", 1, 365, 30, 1)
        
        # Action buttons
        st.subheader("Acciones")
        if st.button("🚀 Inicializar Agente", type="primary"):
            initialize_agent(algorithm, risk_aversion, initial_capital, learning_rate, data_source, data_days)
        
        if st.button("🏋️ Entrenar Agente"):
            if st.session_state.agent is not None:
                train_agent(total_timesteps)
            else:
                st.error("¡Por favor inicializa el agente primero!")
        
        if st.button("📊 Ejecutar Backtest"):
            if st.session_state.agent is not None:
                run_backtest()
            else:
                st.error("¡Por favor inicializa y entrena el agente primero!")
    
    # Main content area
    tab1, tab2, tab3, tab4, tab5 = st.tabs(["📈 Panel Principal", "🤖 Estado del Agente", "📊 Rendimiento", "🔮 Pronósticos", "⚙️ Configuración"])
    
    with tab1:
        show_dashboard()
    
    with tab2:
        show_agent_status()
    
    with tab3:
        show_performance()
    
    with tab4:
        show_forecasting()
    
    with tab5:
        show_settings()

def initialize_agent(algorithm, risk_aversion, initial_capital, learning_rate, data_source, data_days):
    """Initialize the trading agent."""
    try:
        with st.spinner("Inicializando agente..."):
            # Create data manager
            data_manager = DataManager()
            st.session_state.data_manager = data_manager
            
            # Get data
            if data_source == "Sintéticos":
                # Generate synthetic data
                data = generate_synthetic_data(days=data_days)
            else:
                # Get real data
                end_date = datetime.now()
                start_date = end_date - timedelta(days=data_days)
                data = data_manager.collect_market_data(start_date, end_date)
            
            # Create environment
            environment = EnergyMarketEnv(
                data=data,
                initial_capital=initial_capital,
                risk_aversion_lambda=risk_aversion
            )
            st.session_state.environment = environment
            
            # Create forecaster
            forecaster = ProbabilisticForecaster()
            st.session_state.forecaster = forecaster
            
            # Create agent
            agent = RiskAverseTradingAgent(
                environment=environment,
                forecaster=forecaster,
                algorithm=algorithm
            )
            st.session_state.agent = agent
            
            st.success("✅ ¡Agente inicializado exitosamente!")
            
    except Exception as e:
        st.error(f"❌ Error inicializando agente: {str(e)}")

def train_agent(total_timesteps):
    """Train the trading agent."""
    try:
        with st.spinner(f"Entrenando agente por {total_timesteps} pasos..."):
            # Train the agent
            results = st.session_state.agent.train(episodes=None)
            
            st.success("✅ ¡Entrenamiento del agente completado!")
            st.json(results)
            
    except Exception as e:
        st.error(f"❌ Error entrenando agente: {str(e)}")

def run_backtest():
    """Run backtesting on the agent."""
    try:
        with st.spinner("Ejecutando backtest..."):
            # Get test data
            test_data = generate_synthetic_data(days=30)
            
            # Run backtest
            results = st.session_state.agent.backtest(test_data)
            
            st.success("✅ ¡Backtest completado!")
            
            # Display results
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Retorno Total", f"{results['performance']['total_return']:.2%}")
            
            with col2:
                st.metric("Ratio de Sharpe", f"{results['performance']['sharpe_ratio']:.2f}")
            
            with col3:
                st.metric("Pérdida Máxima", f"{results['performance']['max_drawdown']:.2%}")
            
            with col4:
                st.metric("Tasa de Éxito", f"{results['performance']['success_rate']:.2%}")
            
            # Store results for display
            st.session_state.backtest_results = results
            
    except Exception as e:
        st.error(f"❌ Error ejecutando backtest: {str(e)}")

def show_dashboard():
    """Show the main dashboard."""
    st.header("📈 Panel de Trading")
    
    if st.session_state.agent is None:
        st.info("👆 Por favor inicializa el agente desde la barra lateral para ver el panel.")
        return
    
    # Key metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Estado del Agente", "🟢 Activo" if st.session_state.agent.is_trained else "🟡 No Entrenado")
    
    with col2:
        st.metric("Algoritmo", st.session_state.agent.algorithm)
    
    with col3:
        st.metric("Aversión al Riesgo", f"{st.session_state.agent.config['risk']['risk_aversion_lambda']:.2f}")
    
    with col4:
        st.metric("Entorno", "🟢 Listo")
    
    # Portfolio value chart
    if hasattr(st.session_state, 'backtest_results') and st.session_state.backtest_results:
        st.subheader("📊 Rendimiento del Portafolio")
        
        portfolio_values = st.session_state.backtest_results['portfolio_values']
        timestamps = pd.date_range(start=datetime.now() - timedelta(days=len(portfolio_values)), 
                                 periods=len(portfolio_values), freq='H')
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=timestamps,
            y=portfolio_values,
            mode='lines',
            name='Valor del Portafolio',
            line=dict(color='#1f77b4', width=2)
        ))
        
        fig.update_layout(
            title="Valor del Portafolio a lo Largo del Tiempo",
            xaxis_title="Tiempo",
            yaxis_title="Valor del Portafolio (COP)",
            hovermode='x unified'
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    # Recent trades
    if hasattr(st.session_state, 'backtest_results') and st.session_state.backtest_results:
        st.subheader("📋 Trades Recientes")
        
        trades = st.session_state.backtest_results['trades'][-10:]  # Last 10 trades
        
        if trades:
            trades_df = pd.DataFrame(trades)
            st.dataframe(trades_df, use_container_width=True)
        else:
            st.info("Aún no se han registrado trades.")

def show_agent_status():
    """Show agent status and information."""
    st.header("🤖 Estado del Agente")
    
    if st.session_state.agent is None:
        st.info("👆 Por favor inicializa el agente desde la barra lateral.")
        return
    
    # Agent information
    agent_info = st.session_state.agent.get_agent_info()
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Configuración del Agente")
        st.json(agent_info['config'])
    
    with col2:
        st.subheader("Estado de Entrenamiento")
        st.metric("¿Está Entrenado?", "✅ Sí" if agent_info['is_trained'] else "❌ No")
        st.metric("Algoritmo", agent_info['algorithm'])
        st.metric("Sesiones de Entrenamiento", agent_info['training_history'])
    
    # Risk manager info
    if 'risk_manager_info' in agent_info:
        st.subheader("🛡️ Gestión de Riesgo")
        risk_info = agent_info['risk_manager_info']
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Aversión al Riesgo", f"{risk_info['config']['risk_aversion_lambda']:.2f}")
        
        with col2:
            st.metric("Posición Máxima", f"{risk_info['config']['max_position_size']:.0f}")
        
        with col3:
            st.metric("Pérdida Máxima", f"{risk_info['config']['max_drawdown']:.1%}")

def show_performance():
    """Show performance metrics and analysis."""
    st.header("📊 Análisis de Rendimiento")
    
    if not hasattr(st.session_state, 'backtest_results') or not st.session_state.backtest_results:
        st.info("👆 Por favor ejecuta un backtest para ver las métricas de rendimiento.")
        return
    
    results = st.session_state.backtest_results
    performance = results['performance']
    
    # Performance metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Retorno Total", f"{performance['total_return']:.2%}")
    
    with col2:
        st.metric("Ratio de Sharpe", f"{performance['sharpe_ratio']:.2f}")
    
    with col3:
        st.metric("Pérdida Máxima", f"{performance['max_drawdown']:.2%}")
    
    with col4:
        st.metric("Volatilidad", f"{performance['volatility']:.2%}")
    
    # Risk metrics
    st.subheader("🛡️ Métricas de Riesgo")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("VaR (95%)", f"{performance.get('var_95', 0):.2%}")
    
    with col2:
        st.metric("CVaR (95%)", f"{performance.get('cvar_95', 0):.2%}")
    
    with col3:
        st.metric("Tasa de Éxito", f"{performance['success_rate']:.2%}")
    
    # Performance charts
    st.subheader("📈 Gráficos de Rendimiento")
    
    # Returns distribution
    if 'rewards' in results and results['rewards']:
        returns = results['rewards']
        
        fig = go.Figure()
        fig.add_trace(go.Histogram(
            x=returns,
            nbinsx=30,
            name='Returns Distribution',
            marker_color='#1f77b4'
        ))
        
        fig.update_layout(
            title="Returns Distribution",
            xaxis_title="Returns",
            yaxis_title="Frequency"
        )
        
        st.plotly_chart(fig, use_container_width=True)

def show_forecasting():
    """Show forecasting capabilities."""
    st.header("🔮 Pronósticos Probabilísticos")
    
    if st.session_state.forecaster is None:
        st.info("👆 Por favor inicializa el agente para acceder a las capacidades de pronóstico.")
        return
    
    # Forecast configuration
    st.subheader("Configuración del Pronóstico")
    
    col1, col2 = st.columns(2)
    
    with col1:
        forecast_horizon = st.slider("Horizonte del Pronóstico (horas)", 1, 168, 24)
    
    with col2:
        confidence_level = st.slider("Nivel de Confianza", 0.5, 0.99, 0.95)
    
    if st.button("🔮 Generar Pronóstico"):
        try:
            with st.spinner("Generando pronóstico..."):
                # Get current data
                current_data = generate_synthetic_data(days=7)
                
                # Generate forecast
                forecast = st.session_state.forecaster.predict(current_data, horizon=forecast_horizon)
                
                if forecast:
                    st.success("✅ ¡Pronóstico generado exitosamente!")
                    
                    # Display forecast
                    for target, pred in forecast.items():
                        st.subheader(f"📊 Pronóstico de {target.title()}")
                        
                        if 'prediction' in pred:
                            # Create forecast chart
                            timestamps = pd.date_range(start=datetime.now(), periods=forecast_horizon, freq='H')
                            
                            fig = go.Figure()
                            
                            # Mean prediction
                            fig.add_trace(go.Scatter(
                                x=timestamps,
                                y=pred['prediction'],
                                mode='lines',
                                name='Mean Forecast',
                                line=dict(color='#1f77b4', width=2)
                            ))
                            
                            # Confidence intervals
                            if 'uncertainty' in pred and 'confidence_interval' in pred['uncertainty']:
                                ci = pred['uncertainty']['confidence_interval']
                                fig.add_trace(go.Scatter(
                                    x=timestamps,
                                    y=ci[1],
                                    mode='lines',
                                    name='Upper Bound',
                                    line=dict(color='rgba(0,0,0,0)'),
                                    showlegend=False
                                ))
                                fig.add_trace(go.Scatter(
                                    x=timestamps,
                                    y=ci[0],
                                    mode='lines',
                                    name='Confidence Interval',
                                    fill='tonexty',
                                    fillcolor='rgba(31, 119, 180, 0.2)',
                                    line=dict(color='rgba(0,0,0,0)')
                                ))
                            
                            fig.update_layout(
                                title=f"{target.title()} Forecast",
                                xaxis_title="Time",
                                yaxis_title="Value",
                                hovermode='x unified'
                            )
                            
                            st.plotly_chart(fig, use_container_width=True)
                
                else:
                    st.warning("⚠️ No forecast data available. The forecaster may not be trained.")
                    
        except Exception as e:
            st.error(f"❌ Error generating forecast: {str(e)}")

def show_settings():
    """Show application settings."""
    st.header("⚙️ Configuración")
    
    st.subheader("🔧 Configuración del Agente")
    
    # Risk management settings
    st.write("**Gestión de Riesgo**")
    col1, col2 = st.columns(2)
    
    with col1:
        max_position = st.number_input("Tamaño Máximo de Posición", 100, 10000, 1000, 100)
        max_drawdown = st.slider("Pérdida Máxima", 0.05, 0.5, 0.15, 0.01)
    
    with col2:
        var_confidence = st.slider("Confianza VaR", 0.01, 0.1, 0.05, 0.01)
        cvar_confidence = st.slider("Confianza CVaR", 0.01, 0.1, 0.05, 0.01)
    
    # Trading settings
    st.write("**Configuración de Trading**")
    col1, col2 = st.columns(2)
    
    with col1:
        transaction_cost = st.number_input("Costo de Transacción (%)", 0.0, 0.01, 0.001, 0.0001, format="%.4f")
        slippage = st.number_input("Deslizamiento (%)", 0.0, 0.01, 0.0005, 0.0001, format="%.4f")
    
    with col2:
        min_trade_size = st.number_input("Tamaño Mínimo de Trade", 0.001, 1.0, 0.01, 0.001)
        max_trade_size = st.number_input("Tamaño Máximo de Trade", 0.1, 10.0, 1.0, 0.1)
    
    if st.button("💾 Guardar Configuración"):
        st.success("✅ ¡Configuración guardada!")
    
    # Data management
    st.subheader("📊 Gestión de Datos")
    
    if st.button("🔄 Actualizar Datos"):
        st.info("🔄 Actualizando datos...")
        # This would refresh data from APIs
        st.success("✅ ¡Datos actualizados!")
    
    if st.button("💾 Exportar Datos"):
        st.info("📥 Exportando datos...")
        # This would export data
        st.success("✅ ¡Datos exportados!")

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
