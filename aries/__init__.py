"""
Aries: Risk-Averse Trading Agent for Colombian Energy Market

A sophisticated Reinforcement Learning-based trading system for the Colombian
wholesale energy market (XM) with probabilistic forecasting and risk management.
"""

__version__ = "1.0.0"
__author__ = "Aries Team"
__email__ = "aries@example.com"

from .agent import RiskAverseTradingAgent
from .environment import EnergyMarketEnv
from .forecaster import ProbabilisticForecaster
from .data import DataManager
from .agent.risk_manager import RiskManager

__all__ = [
    "RiskAverseTradingAgent",
    "EnergyMarketEnv", 
    "ProbabilisticForecaster",
    "DataManager",
    "RiskManager"
]
