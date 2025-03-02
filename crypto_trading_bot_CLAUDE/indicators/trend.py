# indicators/trend.py
"""
Indicateurs de tendance
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Union

def calculate_ema(df: pd.DataFrame, periods: List[int] = [9, 21, 50, 200]) -> Dict[str, pd.Series]:
    """
    Calcule les moyennes mobiles exponentielles (EMA)
    
    Args:
        df: DataFrame avec les données OHLCV
        periods: Périodes pour les EMA
        
    Returns:
        Dictionnaire des EMA calculées
    """
    result = {}
    
    for period in periods:
        if len(df) >= period:
            result[f'ema_{period}'] = df['close'].ewm(span=period, adjust=False).mean()
        else:
            # Si les données sont insuffisantes, créer une série avec des NaN
            result[f'ema_{period}'] = pd.Series(np.nan, index=df.index)
    
    return result

def calculate_adx(df: pd.DataFrame, period: int = 14) -> Dict[str, pd.Series]:
    """
    Calcule l'Average Directional Index (ADX) - Version optimisée
    
    Args:
        df: DataFrame avec les données OHLCV
        period: Période pour le calcul de l'ADX
        
    Returns:
        Dictionnaire avec ADX, +DI et -DI
    """
    if len(df) < period + 1:
        # Données insuffisantes
        empty_series = pd.Series(np.nan, index=df.index)
        return {
            'adx': empty_series,
            'plus_di': empty_series,
            'minus_di': empty_series
        }
    
    # Créer une copie pour éviter de modifier l'original
    df = df.copy()
    
    # Calcul des True Range (TR)
    df['high_low'] = df['high'] - df['low']
    df['high_close'] = abs(df['high'] - df['close'].shift(1))
    df['low_close'] = abs(df['low'] - df['close'].shift(1))
    df['tr'] = df[['high_low', 'high_close', 'low_close']].max(axis=1)
    
    # Utiliser EWM au lieu de rolling pour plus d'efficacité
    df['atr'] = df['tr'].ewm(alpha=1/period, min_periods=period).mean()
    
    # Calcul des mouvements directionnels
    df['up_move'] = df['high'].diff()
    df['down_move'] = -df['low'].diff()
    
    # Calculer +DM et -DM
    df['plus_dm'] = np.where(
        (df['up_move'] > df['down_move']) & (df['up_move'] > 0),
        df['up_move'],
        0
    )
    
    df['minus_dm'] = np.where(
        (df['down_move'] > df['up_move']) & (df['down_move'] > 0),
        df['down_move'],
        0
    )
    
    # Utiliser EWM pour les smoothed DM
    df['plus_dm_smoothed'] = df['plus_dm'].ewm(alpha=1/period, min_periods=period).mean()
    df['minus_dm_smoothed'] = df['minus_dm'].ewm(alpha=1/period, min_periods=period).mean()
    
    # Calculer +DI et -DI
    df['plus_di'] = 100 * df['plus_dm_smoothed'] / df['atr']
    df['minus_di'] = 100 * df['minus_dm_smoothed'] / df['atr']
    
    # Calculer DX
    df['dx'] = 100 * abs(df['plus_di'] - df['minus_di']) / (df['plus_di'] + df['minus_di']).replace(0, np.nan)
    
    # Calculer ADX
    df['adx'] = df['dx'].ewm(alpha=1/period, min_periods=period).mean()
    
    return {
        'adx': df['adx'],
        'plus_di': df['plus_di'],
        'minus_di': df['minus_di']
    }
def calculate_macd(df: pd.DataFrame, fast_period: int = 12, slow_period: int = 26, 
                 signal_period: int = 9) -> Dict[str, pd.Series]:
    """
    Calcule le MACD (Moving Average Convergence Divergence)
    
    Args:
        df: DataFrame avec les données OHLCV
        fast_period: Période pour l'EMA rapide
        slow_period: Période pour l'EMA lente
        signal_period: Période pour la ligne de signal
        
    Returns:
        Dictionnaire avec MACD, signal et histogramme
    """
    if len(df) < slow_period:
        # Données insuffisantes
        empty_series = pd.Series(np.nan, index=df.index)
        return {
            'macd': empty_series,
            'signal': empty_series,
            'histogram': empty_series
        }
    
    # Calcul des EMA
    fast_ema = df['close'].ewm(span=fast_period, adjust=False).mean()
    slow_ema = df['close'].ewm(span=slow_period, adjust=False).mean()
    
    # Calcul du MACD
    macd = fast_ema - slow_ema
    
    # Calcul de la ligne de signal
    signal = macd.ewm(span=signal_period, adjust=False).mean()
    
    # Calcul de l'histogramme
    histogram = macd - signal
    
    return {
        'macd': macd,
        'signal': signal,
        'histogram': histogram
    }

def detect_trend(df: pd.DataFrame, ema_periods: List[int] = [9, 21, 50]) -> Dict:
    """
    Détecte la tendance à partir des EMA
    
    Args:
        df: DataFrame avec les données OHLCV
        ema_periods: Périodes pour les EMA
        
    Returns:
        Dictionnaire avec la tendance détectée
    """
    # Calculer les EMA
    emas = calculate_ema(df, ema_periods)
    
    # S'assurer que toutes les EMA sont disponibles
    if any(emas[f'ema_{p}'].isna().all() for p in ema_periods):
        return {
            'trend': 'unknown',
            'strength': 0,
            'details': {
                'message': 'EMA non disponibles'
            }
        }
    
    # Récupérer les dernières valeurs
    current_price = df['close'].iloc[-1]
    ema_values = {p: emas[f'ema_{p}'].iloc[-1] for p in ema_periods}
    
    # Détection de la tendance
    ema_short = ema_values[ema_periods[0]]
    ema_medium = ema_values[ema_periods[1]]
    ema_long = ema_values[ema_periods[2]]
    
    # Calcul de la force de la tendance en fonction de l'alignement des EMA
    trend_strength = 0
    trend = 'neutral'
    details = {}
    
    # Vérifier l'alignement haussier (EMA courte > EMA moyenne > EMA longue)
    if ema_short > ema_medium > ema_long:
        trend = 'bullish'
        
        # Évaluer la force de la tendance
        price_vs_ema_short = (current_price / ema_short - 1) * 100
        ema_short_vs_medium = (ema_short / ema_medium - 1) * 100
        ema_medium_vs_long = (ema_medium / ema_long - 1) * 100
        
        trend_strength = (price_vs_ema_short + ema_short_vs_medium + ema_medium_vs_long) / 3
        if trend_strength > 2:
            trend_strength = 1.0  # Fort
        elif trend_strength > 1:
            trend_strength = 0.7  # Modéré
        else:
            trend_strength = 0.3  # Faible
            
        details = {
            'price_vs_ema_short': price_vs_ema_short,
            'ema_short_vs_medium': ema_short_vs_medium,
            'ema_medium_vs_long': ema_medium_vs_long
        }
    
    # Vérifier l'alignement baissier (EMA courte < EMA moyenne < EMA longue)
    elif ema_short < ema_medium < ema_long:
        trend = 'bearish'
        
        # Évaluer la force de la tendance
        price_vs_ema_short = (ema_short / current_price - 1) * 100
        ema_short_vs_medium = (ema_medium / ema_short - 1) * 100
        ema_medium_vs_long = (ema_long / ema_medium - 1) * 100
        
        trend_strength = (price_vs_ema_short + ema_short_vs_medium + ema_medium_vs_long) / 3
        if trend_strength > 2:
            trend_strength = 1.0  # Fort
        elif trend_strength > 1:
            trend_strength = 0.7  # Modéré
        else:
            trend_strength = 0.3  # Faible
        
        details = {
            'price_vs_ema_short': price_vs_ema_short,
            'ema_short_vs_medium': ema_short_vs_medium,
            'ema_medium_vs_long': ema_medium_vs_long
        }
    
    # Tendance neutre ou en transition
    else:
        # Vérifier le croisement des EMA
        if ema_short > ema_medium and ema_medium < ema_long:
            trend = 'potentially_bullish'  # Possible*
            trend = 'potentially_bullish'  # Possible renversement haussier
            trend_strength = 0.2
        elif ema_short < ema_medium and ema_medium > ema_long:
            trend = 'potentially_bearish'  # Possible renversement baissier
            trend_strength = 0.2
        else:
            if current_price > ema_long:
                trend = 'weak_bullish'
                trend_strength = 0.1
            elif current_price < ema_long:
                trend = 'weak_bearish'
                trend_strength = 0.1
            else:
                trend = 'neutral'
                trend_strength = 0
    
    return {
        'trend': trend,
        'strength': trend_strength,
        'details': details
    }
