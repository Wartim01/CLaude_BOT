# indicators/volatility.py
"""
Indicateurs de volatilité
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Union

def calculate_bollinger_bands(df: pd.DataFrame, period: int = 20, 
                           std_dev: float = 2.0) -> Dict[str, pd.Series]:
    """
    Calcule les bandes de Bollinger
    
    Args:
        df: DataFrame avec les données OHLCV
        period: Période pour la moyenne mobile
        std_dev: Nombre d'écarts-types pour les bandes
        
    Returns:
        Dictionnaire avec les bandes supérieure, moyenne et inférieure
    """
    if len(df) < period:
        empty_series = pd.Series(np.nan, index=df.index)
        return {
            'upper': empty_series,
            'middle': empty_series,
            'lower': empty_series,
            'bandwidth': empty_series,
            'percent_b': empty_series
        }
    
    # Calculer la moyenne mobile
    middle = df['close'].rolling(window=period).mean()
    
    # Calculer l'écart-type
    rolling_std = df['close'].rolling(window=period).std()
    
    # Calculer les bandes supérieure et inférieure
    upper = middle + (rolling_std * std_dev)
    lower = middle - (rolling_std * std_dev)
    
    # Calculer la largeur des bandes (bandwidth)
    bandwidth = (upper - lower) / middle
    
    # Calculer %B (position du prix dans les bandes)
    percent_b = (df['close'] - lower) / (upper - lower)
    
    return {
        'upper': upper,
        'middle': middle,
        'lower': lower,
        'bandwidth': bandwidth,
        'percent_b': percent_b
    }

def calculate_atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """
    Calcule l'Average True Range (ATR)
    
    Args:
        df: DataFrame avec les données OHLCV
        period: Période pour l'ATR
        
    Returns:
        Série pandas avec les valeurs de l'ATR
    """
    if len(df) < period + 1:
        return pd.Series(np.nan, index=df.index)
    
    # Calculer le True Range
    high_low = df['high'] - df['low']
    high_close = np.abs(df['high'] - df['close'].shift())
    low_close = np.abs(df['low'] - df['close'].shift())
    
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    
    # Calculer l'ATR (moyenne mobile exponentielle du TR)
    atr = tr.ewm(alpha=1/period, adjust=False).mean()
    
    return atr

def detect_volatility_squeeze(df: pd.DataFrame, bb_period: int = 20, 
                            kc_period: int = 20, kc_mult: float = 1.5) -> Dict:
    """
    Détecte le 'squeeze' (compression de la volatilité) à l'aide des bandes de Bollinger et du canal de Keltner
    
    Args:
        df: DataFrame avec les données OHLCV
        bb_period: Période pour les bandes de Bollinger
        kc_period: Période pour le canal de Keltner
        kc_mult: Multiplicateur pour le canal de Keltner
        
    Returns:
        Dictionnaire avec la détection du squeeze
    """
    if len(df) < max(bb_period, kc_period) + 1:
        return {
            'squeeze': False,
            'strength': 0,
            'details': {
                'message': 'Données insuffisantes'
            }
        }
    
    # Calculer les bandes de Bollinger
    bb = calculate_bollinger_bands(df, period=bb_period)
    
    # Calculer l'ATR pour le canal de Keltner
    atr = calculate_atr(df, period=kc_period)
    
    # Calculer le canal de Keltner
    kc_middle = df['close'].rolling(window=kc_period).mean()
    kc_upper = kc_middle + (atr * kc_mult)
    kc_lower = kc_middle - (atr * kc_mult)
    
    # Détecter le squeeze (les bandes de Bollinger à l'intérieur du canal de Keltner)
    squeeze = (bb['lower'] > kc_lower) & (bb['upper'] < kc_upper)
    
    # Calculer la force du squeeze (compression)
    # Plus la valeur est faible, plus la compression est forte
    compression_ratio = (bb['upper'] - bb['lower']) / (kc_upper - kc_lower)
    
    # Convertir le ratio en force (0-1, où 1 est le squeeze le plus fort)
    strength = 1 - compression_ratio.fillna(1)
    strength = strength.clip(0, 1)
    
    # Récupérer les valeurs récentes
    current_squeeze = squeeze.iloc[-1]
    current_strength = strength.iloc[-1]
    
    # Vérifier combien de temps le squeeze dure
    if current_squeeze:
        squeeze_duration = squeeze.iloc[-20:].sum()
    else:
        squeeze_duration = 0
    
    return {
        'squeeze': bool(current_squeeze),
        'strength': float(current_strength),
        'duration': int(squeeze_duration),
        'details': {
            'bb_width': float(bb['upper'].iloc[-1] - bb['lower'].iloc[-1]),
            'kc_width': float(kc_upper.iloc[-1] - kc_lower.iloc[-1]),
            'compression_ratio': float(compression_ratio.iloc[-1]),
            'historical_squeezes': squeeze.iloc[-50:].sum()
        }
    }

def calculate_volatility(df: pd.DataFrame, window: int = 20, method: str = 'std') -> pd.Series:
    """
    Calcule la volatilité du marché sur une fenêtre donnée
    
    Args:
        df: DataFrame avec les données OHLCV
        window: Fenêtre de calcul de la volatilité
        method: Méthode de calcul ('std', 'atr', 'parkinson')
        
    Returns:
        Série pandas avec les valeurs de volatilité
    """
    if len(df) < window:
        return pd.Series(np.nan, index=df.index)
    
    if method == 'std':
        # Volatilité basée sur l'écart-type des rendements
        returns = df['close'].pct_change()
        volatility = returns.rolling(window=window).std() * np.sqrt(window)  # Annualiser
        
    elif method == 'atr':
        # Volatilité basée sur l'ATR
        atr = calculate_atr(df, period=window)
        volatility = atr / df['close']  # ATR normalisé par le prix
        
    elif method == 'parkinson':
        # Volatilité de Parkinson (basée sur high-low)
        log_hl = (df['high'] / df['low']).apply(np.log)
        volatility = np.sqrt(log_hl.rolling(window=window).mean() / (4 * np.log(2)))
        
    else:
        raise ValueError(f"Méthode de calcul de volatilité non supportée: {method}")
    
    # Convertir en pourcentage
    volatility = volatility * 100
    
    return volatility

def get_market_volatility_state(df: pd.DataFrame, window: int = 20, 
                             long_window: int = 100) -> Dict:
    """
    Détermine l'état de volatilité du marché (faible, moyen, élevé)
    
    Args:
        df: DataFrame avec les données OHLCV
        window: Fenêtre pour la volatilité actuelle
        long_window: Fenêtre pour la volatilité historique
        
    Returns:
        Dictionnaire avec l'état de volatilité
    """
    if len(df) < max(window, long_window):
        return {
            'volatility_state': 'unknown',
            'current_volatility': None,
            'historical_volatility': None,
            'relative_volatility': None
        }
    
    # Calculer la volatilité actuelle et historique
    current_volatility = calculate_volatility(df, window=window).iloc[-1]
    historical_volatility = calculate_volatility(df, window=long_window).iloc[-1]
    
    # Calculer la volatilité relative (par rapport à la moyenne historique)
    if historical_volatility > 0:
        relative_volatility = current_volatility / historical_volatility
    else:
        relative_volatility = 1.0
    
    # Déterminer l'état de volatilité
    if relative_volatility < 0.7:
        volatility_state = 'low'
    elif relative_volatility > 1.5:
        volatility_state = 'high'
    else:
        volatility_state = 'normal'
    
    return {
        'volatility_state': volatility_state,
        'current_volatility': float(current_volatility),
        'historical_volatility': float(historical_volatility),
        'relative_volatility': float(relative_volatility)
    }