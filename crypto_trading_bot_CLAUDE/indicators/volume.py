# indicators/volume.py
"""
Indicateurs de volume
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Union

def calculate_obv(df: pd.DataFrame) -> pd.Series:
    """
    Calcule l'On-Balance Volume (OBV)
    
    Args:
        df: DataFrame avec les données OHLCV
        
    Returns:
        Série pandas avec les valeurs de l'OBV
    """
    if df.empty:
        return pd.Series(dtype=float)
    
    obv = pd.Series(index=df.index, dtype=float)
    obv.iloc[0] = 0
    
    for i in range(1, len(df)):
        if df['close'].iloc[i] > df['close'].iloc[i-1]:
            obv.iloc[i] = obv.iloc[i-1] + df['volume'].iloc[i]
        elif df['close'].iloc[i] < df['close'].iloc[i-1]:
            obv.iloc[i] = obv.iloc[i-1] - df['volume'].iloc[i]
        else:
            obv.iloc[i] = obv.iloc[i-1]
    
    return obv

def calculate_vwap(df: pd.DataFrame) -> pd.Series:
    """
    Calcule le Volume-Weighted Average Price (VWAP)
    
    Args:
        df: DataFrame avec les données OHLCV
        
    Returns:
        Série pandas avec les valeurs du VWAP
    """
    if df.empty:
        return pd.Series(dtype=float)
    
    # Calculer le prix typique
    df = df.copy()
    df['typical_price'] = (df['high'] + df['low'] + df['close']) / 3
    
    # Calculer le produit du prix typique et du volume
    df['tp_volume'] = df['typical_price'] * df['volume']
    
    # Calculer les sommes cumulatives
    df['cum_tp_volume'] = df['tp_volume'].cumsum()
    df['cum_volume'] = df['volume'].cumsum()
    
    # Calculer le VWAP
    vwap = df['cum_tp_volume'] / df['cum_volume']
    
    return vwap

def detect_volume_spike(df: pd.DataFrame, periods: int = 14, 
                      threshold: float = 2.0) -> Dict:
    """
    Détecte les pics de volume
    
    Args:
        df: DataFrame avec les données OHLCV
        periods: Nombre de périodes pour la moyenne
        threshold: Seuil pour détecter un pic (multiplicateur de la moyenne)
        
    Returns:
        Dictionnaire avec la détection de pic de volume
    """
    if len(df) < periods + 1:
        return {
            'spike': False,
            'ratio': 0,
            'details': {
                'message': 'Données insuffisantes'
            }
        }
    
    # Calculer la moyenne mobile du volume
    volume_ma = df['volume'].rolling(window=periods).mean()
    
    # Calculer le ratio du volume actuel par rapport à la moyenne
    current_volume = df['volume'].iloc[-1]
    current_volume_ma = volume_ma.iloc[-1]
    
    volume_ratio = current_volume / current_volume_ma if current_volume_ma > 0 else 0
    
    # Détecter un pic de volume
    is_spike = volume_ratio > threshold
    
    # Vérifier si le pic est associé à une hausse ou à une baisse
    price_change = df['close'].iloc[-1] - df['open'].iloc[-1]
    is_bullish = price_change > 0
    
    return {
        'spike': bool(is_spike),
        'ratio': float(volume_ratio),
        'threshold': float(threshold),
        'bullish': bool(is_bullish) if is_spike else None,
        'details': {
            'current_volume': float(current_volume),
            'average_volume': float(current_volume_ma),
            'price_change': float(price_change),
            'price_change_percent': float(price_change / df['open'].iloc[-1] * 100) if df['open'].iloc[-1] > 0 else 0
        }
    }

def detect_volume_climax(df: pd.DataFrame, periods: int = 14, 
                       threshold: float = 3.0) -> Dict:
    """
    Détecte un climax de volume (volume très élevé associé à une forte variation de prix)
    
    Args:
        df: DataFrame avec les données OHLCV
        periods: Nombre de périodes pour la moyenne
        threshold: Seuil pour détecter un climax (multiplicateur de la moyenne)
        
    Returns:
        Dictionnaire avec la détection de climax de volume
    """
    if len(df) < periods + 1:
        return {
            'climax': False,
            'type': None,
            'details': {
                'message': 'Données insuffisantes'
            }
        }
    
    # Vérifier si c'est un pic de volume
    spike_result = detect_volume_spike(df, periods, threshold)
    
    if not spike_result['spike']:
        return {
            'climax': False,
            'type': None,
            'details': {
                'message': 'Pas de pic de volume',
                'spike_ratio': spike_result['ratio']
            }
        }
    
    # Calculer la taille de la bougie
    body_size = abs(df['close'].iloc[-1] - df['open'].iloc[-1])
    body_size_percent = body_size / df['open'].iloc[-1] * 100 if df['open'].iloc[-1] > 0 else 0
    
    # Calculer la moyenne des tailles de bougie
    body_sizes = abs(df['close'] - df['open'])
    avg_body_size_percent = (body_sizes / df['open']) * 100
    avg_body_size_percent = avg_body_size_percent.replace([np.inf, -np.inf], np.nan).dropna()
    
    avg_body_size_percent = avg_body_size_percent.rolling(window=periods).mean()
    current_avg_body_size_percent = avg_body_size_percent.iloc[-1] if not avg_body_size_percent.empty else 0
    
    # Vérifier si la taille de la bougie est significative
    body_ratio = body_size_percent / current_avg_body_size_percent if current_avg_body_size_percent > 0 else 0
    significant_body = body_ratio > 1.5
    
    # Déterminer le type de climax
    is_bullish = df['close'].iloc[-1] > df['open'].iloc[-1]
    
    if significant_body and is_bullish:
        climax_type = 'buying_climax'
    elif significant_body and not is_bullish:
        climax_type = 'selling_climax'
    else:
        climax_type = 'volume_climax'
    
    return {
        'climax': bool(significant_body),
        'type': climax_type,
        'details': {
            'spike_ratio': float(spike_result['ratio']),
            'body_size_percent': float(body_size_percent),
            'avg_body_size_percent': float(current_avg_body_size_percent),
            'body_ratio': float(body_ratio),
            'is_bullish': bool(is_bullish)
        }
    }

def detect_volume_divergence(df: pd.DataFrame, periods: int = 14) -> Dict:
    """
    Détecte les divergences entre le prix et le volume
    
    Args:
        df: DataFrame avec les données OHLCV
        periods: Nombre de périodes à analyser
        
    Returns:
        Dictionnaire avec la détection de divergence
    """
    if len(df) < periods + 1:
        return {
            'divergence': False,
            'type': None,
            'details': {
                'message': 'Données insuffisantes'
            }
        }
    
    # Récupérer les données récentes
    recent_df = df.iloc[-periods:].copy()
    
    # Calculer les variations en pourcentage
    recent_df['price_change'] = recent_df['close'].pct_change()
    recent_df['volume_change'] = recent_df['volume'].pct_change()
    
    # Ignorer la première ligne (NaN)
    recent_df = recent_df.iloc[1:]
    
    # Compter les cas où les variations de prix et de volume vont dans des directions opposées
    opposite_directions = ((recent_df['price_change'] > 0) & (recent_df['volume_change'] < 0)) | \
                         ((recent_df['price_change'] < 0) & (recent_df['volume_change'] > 0))
    
    divergence_count = opposite_directions.sum()
    divergence_percent = divergence_count / len(recent_df) * 100
    
    # Vérifier les 5 dernières périodes
    recent_opposite = opposite_directions.iloc[-5:].sum()
    recent_percent = recent_opposite / 5 * 100
    
    # Détecter une divergence significative
    significant_divergence = recent_percent > 60
    
    # Déterminer le type de divergence
    recent_price_trend = recent_df['close'].iloc[-1] > recent_df['close'].iloc[0]
    recent_volume_trend = recent_df['volume'].iloc[-1] > recent_df['volume'].iloc[0]
    
    if recent_price_trend and not recent_volume_trend:
        divergence_type = 'price_up_volume_down'
    elif not recent_price_trend and recent_volume_trend:
        divergence_type = 'price_down_volume_up'
    else:
        divergence_type = None
    
    return {
        'divergence': bool(significant_divergence),
        'type': divergence_type,
        'details': {
            'divergence_count': int(divergence_count),
            'divergence_percent': float(divergence_percent),
            'recent_divergence_percent': float(recent_percent),
            'price_trend': 'up' if recent_price_trend else 'down',
            'volume_trend': 'up' if recent_volume_trend else 'down'
        }
    }