"""
Module aggregateur pour les indicateurs techniques.
Ce module importe les fonctions de calcul depuis advanced_features, market_metrics,
momentum, trend, volatility et volume et fournit la fonction add_indicators pour enrichir un DataFrame.
"""
import pandas as pd
from .advanced_features import calculate_advanced_features
from .market_metrics import calculate_market_metrics
from .momentum import calculate_rsi, calculate_stochastic
from .trend import calculate_moving_average, calculate_macd
from .volatility import calculate_atr, calculate_bollinger_bands
from .volume import calculate_volume_oscillator

def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    Enrichit le DataFrame avec des indicateurs techniques calculés.
    """
    # Ajouter des fonctionnalités avancées
    df = calculate_advanced_features(df)
    # Ajouter des métriques de marché
    df = calculate_market_metrics(df)
    # Ajouter les indicateurs de momentum
    df["RSI"] = calculate_rsi(df)
    df["Stochastic"] = calculate_stochastic(df)
    # Ajouter les indicateurs de tendance
    df["MA_50"] = calculate_moving_average(df, window=50)
    df["MA_200"] = calculate_moving_average(df, window=200)
    macd, signal, hist = calculate_macd(df)
    df["MACD"] = macd
    df["MACD_Signal"] = signal
    df["MACD_Hist"] = hist
    # Ajouter les indicateurs de volatilité
    df["ATR"] = calculate_atr(df)
    bollinger = calculate_bollinger_bands(df)
    if isinstance(bollinger, tuple) and len(bollinger) == 2:
        df["Bollinger_Upper"], df["Bollinger_Lower"] = bollinger
    # Ajouter l’oscillateur de volume
    df["Volume_Oscillator"] = calculate_volume_oscillator(df)
    return df
