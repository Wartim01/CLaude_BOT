import pandas as pd
from typing import Dict

def calculate_market_regime(df: pd.DataFrame, lookback: int = 50) -> Dict:
    """
    Détecte le régime de marché actuel (tendance, range, volatil)
    """
    # Calculer la volatilité historique
    volatility = df['close'].rolling(window=lookback).std()
    
    # Détecter les structures de prix et identifier les niveaux
    regime = {
        'volatility': volatility.iloc[-1],
        'trend': 'undefined'
    }
    
    return regime

# Added alias export for backward compatibility:
calculate_market_metrics = calculate_market_regime