import pandas as pd
import numpy as np
from typing import Dict

def _simulate_trading(self, data: pd.DataFrame, strategy, initial_capital: float, symbol: str) -> Dict:
    # Utiliser des structures de données plus efficaces
    equity_history = np.zeros(len(data))
    equity_history[0] = initial_capital
    
    # Préallouer les tableaux pour les statistiques
    position_active = np.zeros(len(data), dtype=bool)
    
    # Apply strategy signals
    signals = strategy.generate_signals(data, symbol)
    position_active[signals] = True
    
    # Calculate returns using self and return results
    return {
        "equity": equity_history,
        "positions": position_active
    }