# indicators/momentum.py
"""
Indicateurs de momentum
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Union

def calculate_rsi(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """
    Calcule le Relative Strength Index (RSI)
    
    Args:
        df: DataFrame avec les données OHLCV
        period: Période pour le RSI
        
    Returns:
        Série pandas avec les valeurs du RSI
    """
    if len(df) < period + 1:
        return pd.Series(np.nan, index=df.index)
    
    # Calculer les variations de prix
    delta = df['close'].diff()
    
    # Séparer les gains et les pertes
    gain = delta.copy()
    loss = delta.copy()
    
    gain[gain < 0] = 0
    loss[loss > 0] = 0
    loss = abs(loss)
    
    # Calculer la moyenne des gains et des pertes
    avg_gain = gain.rolling(window=period).mean()
    avg_loss = loss.rolling(window=period).mean()
    
    # Calculer le RS (Relative Strength)
    rs = avg_gain / avg_loss
    
    # Calculer le RSI
    rsi = 100 - (100 / (1 + rs))
    
    return rsi

def calculate_stochastic(df: pd.DataFrame, k_period: int = 14, d_period: int = 3) -> Dict[str, pd.Series]:
    """
    Calcule l'oscillateur stochastique
    
    Args:
        df: DataFrame avec les données OHLCV
        k_period: Période pour %K
        d_period: Période pour %D
        
    Returns:
        Dictionnaire avec %K et %D
    """
    if len(df) < k_period:
        empty_series = pd.Series(np.nan, index=df.index)
        return {
            'k': empty_series,
            'd': empty_series
        }
    
    # Calculer le plus haut et le plus bas sur la période
    low_min = df['low'].rolling(window=k_period).min()
    high_max = df['high'].rolling(window=k_period).max()
    
    # Calculer %K
    k = 100 * ((df['close'] - low_min) / (high_max - low_min))
    
    # Calculer %D (moyenne mobile simple de %K)
    d = k.rolling(window=d_period).mean()
    
    return {
        'k': k,
        'd': d
    }

def detect_divergence(price_df: pd.DataFrame, indicator: pd.Series, 
                    lookback: int = 15, threshold: float = 0.02,
                    min_peak_distance: int = 3) -> Dict:
    """
    Détecte les divergences entre le prix et un indicateur
    
    Args:
        price_df: DataFrame avec les données de prix
        indicator: Série de l'indicateur (RSI, MACD, etc.)
        lookback: Nombre de périodes à analyser
        threshold: Seuil pour déterminer les sommets/creux significatifs
        
    Returns:
        Dictionnaire avec les divergences détectées
    """
    if len(price_df) < lookback or indicator.isna().all():
        return {
            'bullish': False,
            'bearish': False,
            'details': {
                'message': 'Données insuffisantes'
            }
        }
    
    # Récupérer les données récentes
    recent_price = price_df['close'].iloc[-lookback:].values
    recent_indicator = indicator.iloc[-lookback:].values
    
    # Fonction pour trouver les sommets et creux
    def find_peaks(data):
        peaks = []
        valleys = []
        
        for i in range(1, len(data) - 1):
            if data[i] > data[i-1] and data[i] > data[i+1]:
                if i > 0 and i < len(data) - 1:
                    peaks.append(i)
            elif data[i] < data[i-1] and data[i] < data[i+1]:
                if i > 0 and i < len(data) - 1:
                    valleys.append(i)
        
        return peaks, valleys
    
    # Trouver les sommets et creux
    price_peaks, price_valleys = find_peaks(recent_price)
    indicator_peaks, indicator_valleys = find_peaks(recent_indicator)
    
    # Filtrer les sommets/creux non significatifs
    def is_significant(data, peaks, threshold):
        significant = []
        for p in peaks:
            max_nearby = max(data[max(0, p-2):min(len(data), p+3)])
            if data[p] > max_nearby * (1 - threshold):
                significant.append(p)
        return significant
    
    price_peaks = is_significant(recent_price, price_peaks, threshold)
    price_valleys = is_significant(recent_price, price_valleys, threshold)
    indicator_peaks = is_significant(recent_indicator, indicator_peaks, threshold)
    indicator_valleys = is_significant(recent_indicator, indicator_valleys, threshold)
    
    # Vérifier les divergences
    bullish_divergence = False
    bearish_divergence = False
    details = {}
    
    # Divergence haussière: prix fait un creux plus bas, indicateur fait un creux plus haut
    if len(price_valleys) >= 2 and len(indicator_valleys) >= 2:
        if recent_price[price_valleys[-1]] < recent_price[price_valleys[-2]] and \
           recent_indicator[indicator_valleys[-1]] > recent_indicator[indicator_valleys[-2]]:
            bullish_divergence = True
            details['bullish'] = {
                'price_valley1': price_valleys[-2],
                'price_valley2': price_valleys[-1],
                'indicator_valley1': indicator_valleys[-2],
                'indicator_valley2': indicator_valleys[-1]
            }
    
    # Divergence baissière: prix fait un sommet plus haut, indicateur fait un sommet plus bas
    if len(price_peaks) >= 2 and len(indicator_peaks) >= 2:
        if recent_price[price_peaks[-1]] > recent_price[price_peaks[-2]] and \
           recent_indicator[indicator_peaks[-1]] < recent_indicator[indicator_peaks[-2]]:
            bearish_divergence = True
            details['bearish'] = {
                'price_peak1': price_peaks[-2],
                'price_peak2': price_peaks[-1],
                'indicator_peak1': indicator_peaks[-2],
                'indicator_peak2': indicator_peaks[-1]
            }
    
    return {
        'bullish': bullish_divergence,
        'bearish': bearish_divergence,
        'details': details
    }
