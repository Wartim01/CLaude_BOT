# indicators/advanced_features.py
"""
Module pour la création de caractéristiques avancées pour les modèles LSTM
Inclut des indicateurs techniques complexes, des mesures de sentiment et des patterns
"""
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Union, Tuple
import talib
from scipy import stats

from indicators.trend import calculate_ema, calculate_adx, calculate_macd
from indicators.momentum import calculate_rsi, calculate_stochastic
from indicators.volatility import calculate_bollinger_bands, calculate_atr
from indicators.volume import calculate_obv, calculate_vwap
from utils.logger import setup_logger

logger = setup_logger("advanced_features")

def calculate_advanced_features(data: pd.DataFrame, include_all: bool = False) -> pd.DataFrame:
    """
    Calcule des caractéristiques avancées pour l'entraînement du modèle
    
    Args:
        data: DataFrame OHLCV
        include_all: Inclure toutes les caractéristiques (potentiellement lent)
        
    Returns:
        DataFrame avec les caractéristiques avancées
    """
    if data.empty:
        logger.warning("DataFrame vide, impossible de calculer les caractéristiques avancées")
        return pd.DataFrame()
    
    # Copier les données d'entrée pour éviter les modifications
    df = data.copy()
    
    # S'assurer que les colonnes OHLCV sont présentes
    required_cols = ['open', 'high', 'low', 'close', 'volume']
    missing_cols = [col for col in required_cols if col not in df.columns]
    
    if missing_cols:
        logger.error(f"Colonnes manquantes: {missing_cols}")
        return pd.DataFrame()
    
    try:
        # 1. Indicateurs de tendance
        add_trend_features(df)
        
        # 2. Indicateurs de momentum
        add_momentum_features(df)
        
        # 3. Indicateurs de volatilité
        add_volatility_features(df)
        
        # 4. Indicateurs de volume
        add_volume_features(df)
        
        # 5. Indicateurs de support/résistance
        add_support_resistance_features(df)
        
        # 6. Caractéristiques temporelles cycliques
        add_time_features(df)
        
        # 7. Caractéristiques des bougies (patterns)
        add_candlestick_patterns(df)
        
        # 8. Fonctionnalités avancées basées sur la microstructure du marché
        if include_all:
            add_market_microstructure_features(df)
        
        # 9. Caractéristiques de divergence
        add_divergence_features(df)
        
        # 10. Caractéristiques fractales
        if include_all:
            add_fractal_features(df)
        
        # Nettoyer les NaN
        df.fillna(method='ffill', inplace=True)
        df.fillna(method='bfill', inplace=True)
        df.fillna(0, inplace=True)
        
        return df
    
    except Exception as e:
        logger.error(f"Erreur lors du calcul des caractéristiques avancées: {str(e)}")
        # Renvoyer les données originales si une erreur se produit
        return data


def add_trend_features(df: pd.DataFrame) -> None:
    """
    Ajoute des caractéristiques basées sur la tendance
    
    Args:
        df: DataFrame OHLCV (modifié en place)
    """
    # EMAs à différentes périodes
    ema_periods = [9, 21, 50, 100, 200]
    emas = calculate_ema(df, ema_periods)
    
    for period, ema_series in emas.items():
        df[f'ema_{period}'] = ema_series
    
    # Distances relatives aux EMAs
    for period in ema_periods:
        df[f'dist_to_ema_{period}'] = (df['close'] - df[f'ema_{period}']) / df[f'ema_{period}'] * 100
    
    # Croisements des EMAs (signaux de tendance)
    # Court terme vs. moyen terme
    df['ema_9_21_cross'] = np.where(df['ema_9'] > df['ema_21'], 1, -1)
    
    # Moyen terme vs. long terme
    df['ema_21_50_cross'] = np.where(df['ema_21'] > df['ema_50'], 1, -1)
    
    # Long terme vs. très long terme
    if 'ema_100' in df.columns and 'ema_200' in df.columns:
        df['ema_100_200_cross'] = np.where(df['ema_100'] > df['ema_200'], 1, -1)
    
    # MACD (Moving Average Convergence Divergence)
    macd_data = calculate_macd(df)
    df['macd'] = macd_data['macd']
    df['macd_signal'] = macd_data['signal']
    df['macd_hist'] = macd_data['histogram']
    
    # ADX (Average Directional Index) pour la force de la tendance
    adx_data = calculate_adx(df)
    df['adx'] = adx_data['adx']
    df['plus_di'] = adx_data['plus_di']
    df['minus_di'] = adx_data['minus_di']
    
    # Direction de la tendance basée sur l'ADX
    df['trend_strength'] = df['adx']
    df['trend_direction'] = np.where(df['plus_di'] > df['minus_di'], 1, -1)
    
    # Tendance basée sur les prix récents (court terme)
    window = 10
    df['price_trend_10'] = df['close'].rolling(window=window).apply(
        lambda x: stats.linregress(np.arange(len(x)), x)[0], raw=True)
    
    # Normaliser la tendance des prix
    if 'price_trend_10' in df.columns:
        std = df['price_trend_10'].std()
        if std > 0:
            df['price_trend_10_norm'] = df['price_trend_10'] / std


def add_momentum_features(df: pd.DataFrame) -> None:
    """
    Ajoute des caractéristiques basées sur le momentum
    
    Args:
        df: DataFrame OHLCV (modifié en place)
    """
    # RSI (Relative Strength Index)
    df['rsi'] = calculate_rsi(df)
    
    # Zones de surachat/survente du RSI
    df['rsi_overbought'] = np.where(df['rsi'] > 70, 1, 0)
    df['rsi_oversold'] = np.where(df['rsi'] < 30, 1, 0)
    
    # Stochastique
    stoch_data = calculate_stochastic(df)
    df['stoch_k'] = stoch_data['k']
    df['stoch_d'] = stoch_data['d']
    
    # Croisement stochastique
    df['stoch_cross'] = np.where(df['stoch_k'] > df['stoch_d'], 1, -1)
    
    # Rate of Change (ROC) à différentes périodes
    for period in [5, 10, 21, 50]:
        df[f'roc_{period}'] = df['close'].pct_change(period) * 100
    
    # TSI (True Strength Index)
    def calculate_tsi(close, r=25, s=13):
        # Force du mouvement
        m = close.diff()
        # Double lissage exponentiel
        m1 = m.ewm(span=r, adjust=False).mean()
        m2 = m1.ewm(span=s, adjust=False).mean()
        # Valeur absolue de la force du mouvement
        a = abs(m)
        # Double lissage exponentiel
        a1 = a.ewm(span=r, adjust=False).mean()
        a2 = a1.ewm(span=s, adjust=False).mean()
        # TSI
        return (m2 / a2) * 100
    
    df['tsi'] = calculate_tsi(df['close'])
    
    # Momentum sur plusieurs périodes
    for period in [14, 30, 90]:
        df[f'momentum_{period}'] = df['close'] - df['close'].shift(period)
        # Normaliser le momentum
        std = df[f'momentum_{period}'].std()
        if std > 0:
            df[f'momentum_{period}_norm'] = df[f'momentum_{period}'] / std
    
    # Accélération du prix (dérivée seconde)
    df['price_acceleration'] = df['close'].diff().diff()
    
    # CCI (Commodity Channel Index)
    def calculate_cci(high, low, close, period=20):
        tp = (high + low + close) / 3
        ma_tp = tp.rolling(window=period).mean()
        md_tp = tp.rolling(window=period).apply(lambda x: np.abs(x - x.mean()).mean())
        return (tp - ma_tp) / (0.015 * md_tp)
    
    df['cci'] = calculate_cci(df['high'], df['low'], df['close'])


def add_volatility_features(df: pd.DataFrame) -> None:
    """
    Ajoute des caractéristiques basées sur la volatilité
    
    Args:
        df: DataFrame OHLCV (modifié en place)
    """
    # Bandes de Bollinger
    bb_data = calculate_bollinger_bands(df)
    df['bb_upper'] = bb_data['upper']
    df['bb_middle'] = bb_data['middle']
    df['bb_lower'] = bb_data['lower']
    df['bb_width'] = bb_data['bandwidth']
    df['bb_percent_b'] = bb_data['percent_b']
    
    # ATR (Average True Range)
    df['atr'] = calculate_atr(df)
    df['atr_percent'] = df['atr'] / df['close'] * 100  # ATR relatif au prix
    
    # Volatilité historique
    for period in [10, 21, 50]:
        df[f'volatility_{period}'] = df['close'].pct_change().rolling(window=period).std() * np.sqrt(period)
    
    # Keltner Channels
    def calculate_keltner_channels(df, period=20, multiplier=2):
        middle = df['close'].ewm(span=period, adjust=False).mean()
        range_avg = df['atr']
        upper = middle + multiplier * range_avg
        lower = middle - multiplier * range_avg
        return {
            'middle': middle,
            'upper': upper,
            'lower': lower,
            'width': (upper - lower) / middle * 100
        }
    
    kc = calculate_keltner_channels(df)
    df['kc_upper'] = kc['upper']
    df['kc_middle'] = kc['middle']
    df['kc_lower'] = kc['lower']
    df['kc_width'] = kc['width']
    
    # Position par rapport aux Keltner Channels
    df['kc_position'] = (df['close'] - df['kc_lower']) / (df['kc_upper'] - df['kc_lower'])
    
    # Squeeze Momentum (combinaison de Bollinger et Keltner)
    df['squeeze_on'] = np.where((df['bb_lower'] > df['kc_lower']) & (df['bb_upper'] < df['kc_upper']), 1, 0)
    
    # Choppiness Index (marché directionnel vs en range)
    def calculate_choppiness(high, low, close, period=14):
        atr_sum = calculate_atr(df).rolling(window=period).sum()
        range_high_low = high.rolling(window=period).max() - low.rolling(window=period).min()
        return 100 * np.log10(atr_sum / range_high_low) / np.log10(period)
    
    df['choppiness'] = calculate_choppiness(df['high'], df['low'], df['close'])


def add_volume_features(df: pd.DataFrame) -> None:
    """
    Ajoute des caractéristiques basées sur le volume
    
    Args:
        df: DataFrame OHLCV (modifié en place)
    """
    # Volume relatif (par rapport à la moyenne)
    for period in [5, 10, 21, 50]:
        df[f'rel_volume_{period}'] = df['volume'] / df['volume'].rolling(window=period).mean()
    
    # OBV (On-Balance Volume)
    df['obv'] = calculate_obv(df)
    
    # Chaikin Money Flow
    def calculate_cmf(df, period=20):
        mf_vol = ((df['close'] - df['low']) - (df['high'] - df['close'])) / (df['high'] - df['low']) * df['volume']
        return mf_vol.rolling(window=period).sum() / df['volume'].rolling(window=period).sum()
    
    df['cmf'] = calculate_cmf(df)
    
    # Volume Force Index
    df['vfi'] = df['volume'] * df['close'].diff()
    
    # Accélération du volume
    df['volume_diff'] = df['volume'].diff()
    df['volume_diff_pct'] = df['volume'].pct_change() * 100
    
    # Divergence volume/prix
    df['price_up'] = np.where(df['close'] > df['close'].shift(1), 1, 0)
    df['volume_up'] = np.where(df['volume'] > df['volume'].shift(1), 1, 0)
    df['vol_price_divergence'] = np.where(df['price_up'] != df['volume_up'], 1, 0)
    
    # VWAP (Volume-Weighted Average Price)
    df['vwap'] = calculate_vwap(df)
    df['vwap_diff'] = (df['close'] - df['vwap']) / df['vwap'] * 100


def add_support_resistance_features(df: pd.DataFrame) -> None:
    """
    Ajoute des caractéristiques relatives aux niveaux de support et résistance
    
    Args:
        df: DataFrame OHLCV (modifié en place)
    """
    # Identification des pivots (sommets et creux locaux)
    window = 10  # Fenêtre pour identifier les pivots
    
    # Sommets (résistances potentielles)
    df['pivot_high'] = df['high'].rolling(window=window, center=True).apply(
        lambda x: x[len(x)//2] == max(x) and x[len(x)//2] > x[len(x)//2 - 1] and x[len(x)//2] > x[len(x)//2 + 1], 
        raw=True
    ).fillna(0)
    
    # Creux (supports potentiels)
    df['pivot_low'] = df['low'].rolling(window=window, center=True).apply(
        lambda x: x[len(x)//2] == min(x) and x[len(x)//2] < x[len(x)//2 - 1] and x[len(x)//2] < x[len(x)//2 + 1], 
        raw=True
    ).fillna(0)
    
    # Importance des niveaux (nombre de touches)
    level_importance = {}
    
    # Simplifier les prix pour regrouper les niveaux similaires
    def simplify_price(price, precision=2):
        return round(price / 10**precision) * 10**precision
    
    # Trouver les niveaux importants
    for i in range(window, len(df)):
        if df['pivot_high'].iloc[i]:
            level = simplify_price(df['high'].iloc[i])
            level_importance[level] = level_importance.get(level, 0) + 1
        
        if df['pivot_low'].iloc[i]:
            level = simplify_price(df['low'].iloc[i])
            level_importance[level] = level_importance.get(level, 0) + 1
    
    # Identifier les niveaux importants (ceux touchés plusieurs fois)
    important_levels = [level for level, count in level_importance.items() if count >= 2]
    
    # Distance aux niveaux importants
    for i in range(len(df)):
        # Initialiser à une grande valeur
        closest_support = None
        closest_resistance = None
        min_support_distance = float('inf')
        min_resistance_distance = float('inf')
        
        for level in important_levels:
            if level < df['close'].iloc[i]:  # Support
                distance = (df['close'].iloc[i] - level) / level * 100
                if distance < min_support_distance:
                    min_support_distance = distance
                    closest_support = level
            elif level > df['close'].iloc[i]:  # Résistance
                distance = (level - df['close'].iloc[i]) / df['close'].iloc[i] * 100
                if distance < min_resistance_distance:
                    min_resistance_distance = distance
                    closest_resistance = level
        
        # Ajouter aux données
        if i == 0:
            df['distance_to_support'] = np.nan
            df['distance_to_resistance'] = np.nan
        
        if closest_support is not None:
            df.loc[df.index[i], 'distance_to_support'] = min_support_distance
        if closest_resistance is not None:
            df.loc[df.index[i], 'distance_to_resistance'] = min_resistance_distance
    
    # Remplir les valeurs manquantes
    df['distance_to_support'].fillna(method='ffill', inplace=True)
    df['distance_to_resistance'].fillna(method='ffill', inplace=True)
    df['distance_to_support'].fillna(100, inplace=True)  # Valeur par défaut élevée
    df['distance_to_resistance'].fillna(100, inplace=True)  # Valeur par défaut élevée


def add_time_features(df: pd.DataFrame) -> None:
    """
    Ajoute des caractéristiques temporelles cycliques
    
    Args:
        df: DataFrame OHLCV (modifié en place)
    """
    # Vérifier que l'index est un DatetimeIndex
    if not isinstance(df.index, pd.DatetimeIndex):
        logger.warning("Index non temporel, impossible d'ajouter des caractéristiques temporelles")
        return
    
    # Heure de la journée (représentation cyclique)
    df['hour'] = df.index.hour
    df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
    df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
    
    # Jour de la semaine (représentation cyclique)
    df['day_of_week'] = df.index.dayofweek
    df['day_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
    df['day_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7)
    
    # Jour du mois (représentation cyclique)
    df['day_of_month'] = df.index.day
    df['day_of_month_sin'] = np.sin(2 * np.pi * df['day_of_month'] / 31)
    df['day_of_month_cos'] = np.cos(2 * np.pi * df['day_of_month'] / 31)
    
    # Mois de l'année (représentation cyclique)
    df['month'] = df.index.month
    df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
    df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
    
    # Pour les crypto-monnaies, ajouter des indicateurs pour les périodes de forte activité
    # Par exemple, les périodes d'ouverture/fermeture des marchés traditionnels
    df['us_market_open'] = ((df['hour'] >= 9) & (df['hour'] < 16) & 
                            (df['day_of_week'] >= 0) & (df['day_of_week'] <= 4)).astype(int)
    
    df['asia_market_open'] = ((df['hour'] >= 0) & (df['hour'] < 8) & 
                              (df['day_of_week'] >= 0) & (df['day_of_week'] <= 4)).astype(int)
    
    # Supprimer les colonnes intermédiaires pour ne garder que les représentations cycliques
    df.drop(['hour', 'day_of_week', 'day_of_month', 'month'], axis=1, inplace=True, errors='ignore')


def add_candlestick_patterns(df: pd.DataFrame) -> None:
    """
    Ajoute des patterns de chandeliers japonais
    
    Args:
        df: DataFrame OHLCV (modifié en place)
    """
    # Calculer les caractéristiques des bougies
    df['body_size'] = abs(df['close'] - df['open'])
    df['body_size_pct'] = df['body_size'] / df['open'] * 100
    
    df['upper_wick'] = df.apply(lambda x: x['high'] - max(x['open'], x['close']), axis=1)
    df['lower_wick'] = df.apply(lambda x: min(x['open'], x['close']) - x['low'], axis=1)
    
    df['upper_wick_pct'] = df['upper_wick'] / df['open'] * 100
    df['lower_wick_pct'] = df['lower_wick'] / df['open'] * 100
    
    # Patterns de bougies (utilisant talib)
    try:
        # Doji (petite bougie avec grandes mèches)
        df['doji'] = talib.CDLDOJI(df['open'].values, df['high'].values, df['low'].values, df['close'].values)
        
        # Marteau et marteau inversé (indicateurs de retournement)
        df['hammer'] = talib.CDLHAMMER(df['open'].values, df['high'].values, df['low'].values, df['close'].values)
        df['inverted_hammer'] = talib.CDLINVERTEDHAMMER(df['open'].values, df['high'].values, df['low'].values, df['close'].values)
        
        # Patterns d'absorption (indicateurs de retournement)
        df['engulfing'] = talib.CDLENGULFING(df['open'].values, df['high'].values, df['low'].values, df['close'].values)
        
        # Étoiles filantes (indicateurs de retournement baissier)
        df['shooting_star'] = talib.CDLSHOOTINGSTAR(df['open'].values, df['high'].values, df['low'].values, df['close'].values)
        
        # Morning Star et Evening Star (patterns de retournement puissants)
        df['morning_star'] = talib.CDLMORNINGSTAR(df['open'].values, df['high'].values, df['low'].values, df['close'].values)
        df['evening_star'] = talib.CDLEVENINGSTAR(df['open'].values, df['high'].values, df['low'].values, df['close'].values)
        
        # Three White Soldiers et Three Black Crows (patterns de continuation puissants)
        df['three_white_soldiers'] = talib.CDL3WHITESOLDIERS(df['open'].values, df['high'].values, df['low'].values, df['close'].values)
        df['three_black_crows'] = talib.CDL3BLACKCROWS(df['open'].values, df['high'].values, df['low'].values, df['close'].values)
        
        # Créer des indicateurs agrégés de patterns haussiers/baissiers
        bullish_patterns = ['hammer', 'morning_star', 'three_white_soldiers']
        bearish_patterns = ['shooting_star', 'evening_star', 'three_black_crows']
        
        # Pour l'engulfing, il faut vérifier la direction
        df['bullish_engulfing'] = np.where(df['engulfing'] > 0, 1, 0)
        df['bearish_engulfing'] = np.where(df['engulfing'] < 0, 1, 0)
        
        bullish_patterns.append('bullish_engulfing')
        bearish_patterns.append('bearish_engulfing')
        
        # Combiner les patterns
        df['bullish_pattern_count'] = df[bullish_patterns].sum(axis=1)
        df['bearish_pattern_count'] = df[bearish_patterns].sum(axis=1)
        
        # Score global des patterns (positif pour haussier, négatif pour baissier)
        df['pattern_score'] = df['bullish_pattern_count'] - df['bearish_pattern_count']
    
    except (AttributeError, ImportError, Exception) as e:
        logger.warning(f"Impossible de calculer les patterns de bougies: {str(e)}")
        # Créer des colonnes vides pour les patterns
        for col in ['doji', 'hammer', 'inverted_hammer', 'engulfing', 'shooting_star', 
                   'morning_star', 'evening_star', 'three_white_soldiers', 'three_black_crows',
                   'bullish_engulfing', 'bearish_engulfing', 'bullish_pattern_count', 
                   'bearish_pattern_count', 'pattern_score']:
            df[col] = 0


def add_market_microstructure_features(df: pd.DataFrame) -> None:
    """
    Ajoute des caractéristiques basées sur la microstructure du marché
    
    Args:
        df: DataFrame OHLCV (modifié en place)
    """
    # Spread (High-Low range)
    df['hl_range'] = df['high'] - df['low']
    df['hl_range_pct'] = df['hl_range'] / df['close'] * 100
    
    # Prix d'ouverture relatif dans la range High-Low
    df['open_position'] = (df['open'] - df['low']) / df['hl_range']
    
    # Prix de clôture relatif dans la range High-Low
    df['close_position'] = (df['close'] - df['low']) / df['hl_range']
    
    # Mesure de l'imbalance offre/demande (basée sur la position de clôture)
    df['buy_sell_imbalance'] = 2 * df['close_position'] - 1  # Entre -1 (forte vente) et 1 (fort achat)
    
    # Efficacité du mouvement: distance close-to-close vs. range high-low
    df['price_efficiency'] = abs(df['close'] - df['close'].shift(1)) / df['hl_range']
    
    # Caractéristiques d'ordre supérieur (indicateurs de régime de marché)
    # Calculer une métrique de "surprise" de prix
    df['price_surprise'] = abs(df['close'] - df['open']) / df['atr']
    
    # Taux de hausse/baisse (ratio de bougies haussières vs baissières)
    window = 10
    df['up_candle'] = np.where(df['close'] > df['open'], 1, 0)
    df['down_candle'] = np.where(df['close'] < df['open'], 1, 0)
    df['up_ratio'] = df['up_candle'].rolling(window=window).sum() / window
    
    # Volatilité relative de fin de bougie vs. corps
    df['close_volatility_ratio'] = abs(df['close'] - df['close'].shift(1)) / df['body_size']
    df['close_volatility_ratio'].replace([np.inf, -np.inf], 1, inplace=True)


def add_divergence_features(df: pd.DataFrame) -> None:
    """
    Ajoute des caractéristiques de divergence entre prix et indicateurs
    
    Args:
        df: DataFrame OHLCV (modifié en place)
    """
    # Longueur de la fenêtre pour détecter les divergences
    window = 5
    
    # Divergence prix-RSI
    if 'rsi' in df.columns:
        # Calculer les hauts et bas locaux pour le prix et le RSI
        df['price_high'] = df['close'].rolling(window=window, center=True).apply(
            lambda x: 1 if np.argmax(x) == len(x)//2 else 0, raw=True)
        df['price_low'] = df['close'].rolling(window=window, center=True).apply(
            lambda x: 1 if np.argmin(x) == len(x)//2 else 0, raw=True)
        
        df['rsi_high'] = df['rsi'].rolling(window=window, center=True).apply(
            lambda x: 1 if np.argmax(x) == len(x)//2 else 0, raw=True)
        df['rsi_low'] = df['rsi'].rolling(window=window, center=True).apply(
            lambda x: 1 if np.argmin(x) == len(x)//2 else 0, raw=True)
        
        # Divergence baissière: prix en hausse, RSI en baisse
        df['bearish_divergence'] = np.where((df['price_high'] == 1) & (df['rsi_high'] == 0), 1, 0)
        
        # Divergence haussière: prix en baisse, RSI en hausse
        df['bullish_divergence'] = np.where((df['price_low'] == 1) & (df['rsi_low'] == 0), 1, 0)
    
    # Divergence prix-MACD
    if 'macd' in df.columns and 'macd_hist' in df.columns:
        # Divergence basée sur l'histogramme MACD
        df['macd_hist_high'] = df['macd_hist'].rolling(window=window, center=True).apply(
            lambda x: 1 if np.argmax(x) == len(x)//2 else 0, raw=True)
        df['macd_hist_low'] = df['macd_hist'].rolling(window=window, center=True).apply(
            lambda x: 1 if np.argmin(x) == len(x)//2 else 0, raw=True)
        
        # Divergence baissière: prix en hausse, MACD histogramme en baisse
        df['macd_bearish_divergence'] = np.where((df['price_high'] == 1) & (df['macd_hist_high'] == 0), 1, 0)
        
        # Divergence haussière: prix en baisse, MACD histogramme en hausse
        df['macd_bullish_divergence'] = np.where((df['price_low'] == 1) & (df['macd_hist_low'] == 0), 1, 0)
    
    # Divergence prix-volume
    if 'price_high' in df.columns:
        df['volume_high'] = df['volume'].rolling(window=window, center=True).apply(
            lambda x: 1 if np.argmax(x) == len(x)//2 else 0, raw=True)
        
        # Divergence volume-prix: prix en hausse, volume en baisse
        df['vol_price_bearish_div'] = np.where((df['price_high'] == 1) & (df['volume_high'] == 0), 1, 0)
    
    # Supprimer les colonnes intermédiaires
    for col in ['price_high', 'price_low', 'rsi_high', 'rsi_low', 
               'macd_hist_high', 'macd_hist_low', 'volume_high']:
        if col in df.columns:
            df.drop(col, axis=1, inplace=True)


def add_fractal_features(df: pd.DataFrame) -> None:
    """
    Ajoute des caractéristiques fractales et chaotiques
    
    Args:
        df: DataFrame OHLCV (modifié en place)
    """
    # Fractales de Williams (indicateurs de retournement)
    def detect_fractals(high, low, window=5):
        bull_fractal = np.zeros_like(high)
        bear_fractal = np.zeros_like(low)
        
        for i in range(window, len(high) - window):
            # Fractale haussière (low point)
            if all(low[i] < low[i-j] for j in range(1, window+1)) and \
               all(low[i] < low[i+j] for j in range(1, window+1)):
                bull_fractal[i] = 1
            
            # Fractale baissière (high point)
            if all(high[i] > high[i-j] for j in range(1, window+1)) and \
               all(high[i] > high[i+j] for j in range(1, window+1)):
                bear_fractal[i] = 1
        
        return bull_fractal, bear_fractal
    
    # Calculer les fractales
    bull_fractal, bear_fractal = detect_fractals(df['high'].values, df['low'].values)
    df['bull_fractal'] = bull_fractal
    df['bear_fractal'] = bear_fractal
    
    # Dimension fractale du marché (mesure de tendance vs. chaos)
    def fractal_dimension(series, window=100):
        """Implémentation simplifiée de l'exposant de Hurst"""
        if len(series) < window:
            return np.zeros_like(series)
        
        result = np.zeros_like(series)
        
        for i in range(window, len(series)):
            current_series = series[i-window:i]
            # Calculer la plage normalisée
            changes = current_series.diff().dropna()
            dev = changes.std()
            
            if dev == 0:
                result[i] = 0.5  # Random walk
                continue
            
            # Calculer l'écart rescalé
            cumsum = changes.cumsum()
            diff = cumsum - cumsum.mean()
            range_val = diff.max() - diff.min()
            rs = range_val / dev
            
            # Approximation de l'exposant de Hurst (H)
            # H < 0.5: anti-persistent, H = 0.5: random walk, H > 0.5: persistent
            hurst = np.log(rs) / np.log(window)
            result[i] = hurst
            
        return result
    
    # Calculer la dimension fractale du marché
    df['hurst_exponent'] = fractal_dimension(df['close'])
    
    # Classification du régime de marché basée sur l'exposant de Hurst
    df['is_trending'] = np.where(df['hurst_exponent'] > 0.6, 1, 0)
    df['is_mean_reverting'] = np.where(df['hurst_exponent'] < 0.4, 1, 0)
    df['is_random_walk'] = np.where((df['hurst_exponent'] >= 0.4) & (df['hurst_exponent'] <= 0.6), 1, 0)
    
    # Lyapunov exponent (estimation simplifiée) - mesure de prévisibilité/chaos
    def estimate_lyapunov(series, window=50, delay=1):
        """Estimation simplifiée de l'exposant de Lyapunov"""
        if len(series) < window + delay:
            return np.zeros_like(series)
        
        result = np.zeros_like(series)
        
        for i in range(window + delay, len(series)):
            # Créer des vecteurs décalés
            current = series[i-window:i]
            delayed = series[i-window-delay:i-delay]
            
            # Somme logarithmique des différences
            log_diffs = np.log(np.abs(current - delayed) + 1e-10)
            
            # Estimation grossière de l'exposant de Lyapunov
            lyapunov = np.mean(log_diffs) / delay
            result[i] = lyapunov
        
        return result
    
    # Calculer l'exposant de Lyapunov
    df['lyapunov'] = estimate_lyapunov(df['close'])
    df['predictability'] = np.exp(-np.abs(df['lyapunov']))  # Transformation en indice de prévisibilité (0-1)


# Fonctions auxiliaires pour l'ingénierie des caractéristiques avancées

def extract_feature_groups(df: pd.DataFrame) -> Dict[str, List[str]]:
    """
    Extrait les caractéristiques par groupe pour faciliter la sélection
    
    Args:
        df: DataFrame avec les caractéristiques
        
    Returns:
        Dictionnaire des groupes de caractéristiques
    """
    # Initialiser les groupes
    feature_groups = {
        'trend': [],
        'momentum': [],
        'volatility': [],
        'volume': [],
        'support_resistance': [],
        'time': [],
        'candlestick': [],
        'microstructure': [],
        'divergence': [],
        'fractal': [],
        'price': ['open', 'high', 'low', 'close'],
        'other': []
    }
    
    # Classer les caractéristiques par préfixe/mot-clé
    for col in df.columns:
        if col in ['open', 'high', 'low', 'close', 'volume']:
            continue  # Déjà traité
            
        if col.startswith(('ema_', 'dist_to_ema_', 'macd', 'adx', 'trend_', 'price_trend')):
            feature_groups['trend'].append(col)
        elif col.startswith(('rsi', 'stoch_', 'roc_', 'tsi', 'momentum_', 'cci')):
            feature_groups['momentum'].append(col)
        elif col.startswith(('bb_', 'atr', 'volatility_', 'kc_', 'squeeze', 'choppiness')):
            feature_groups['volatility'].append(col)
        elif col.startswith(('rel_volume_', 'obv', 'cmf', 'vfi', 'volume_', 'vwap')):
            feature_groups['volume'].append(col)
        elif col.startswith(('pivot_', 'distance_to_')):
            feature_groups['support_resistance'].append(col)
        elif col.startswith(('hour_', 'day_', 'month_', 'us_market', 'asia_market')):
            feature_groups['time'].append(col)
        elif col.startswith(('body_', 'upper_wick', 'lower_wick', 'doji', 'hammer', 'pattern_')):
            feature_groups['candlestick'].append(col)
        elif col.startswith(('hl_range', 'open_position', 'close_position', 'buy_sell', 'price_efficiency')):
            feature_groups['microstructure'].append(col)
        elif col.startswith(('bullish_div', 'bearish_div', 'macd_bullish', 'macd_bearish', 'vol_price_')):
            feature_groups['divergence'].append(col)
        elif col.startswith(('bull_fractal', 'bear_fractal', 'hurst_', 'is_trending', 'lyapunov')):
            feature_groups['fractal'].append(col)
        else:
            feature_groups['other'].append(col)
    
    return feature_groups


def select_important_features(df: pd.DataFrame, target_var: str = None, top_n: int = 30) -> List[str]:
    """
    Sélectionne les caractéristiques les plus importantes
    
    Args:
        df: DataFrame avec les caractéristiques
        target_var: Variable cible pour l'importance (si None, utilise la direction du prix futur)
        top_n: Nombre de caractéristiques à sélectionner
        
    Returns:
        Liste des caractéristiques les plus importantes
    """
    # Si pas de variable cible, créer une variable de direction du prix (futur)
    if target_var is None:
        df['future_direction'] = np.where(df['close'].shift(-1) > df['close'], 1, 0)
        target_var = 'future_direction'
    
    # Si la cible n'est pas dans le dataframe, retourner une liste vide
    if target_var not in df.columns:
        return []
    
    # Créer un DataFrame pour l'analyse
    data = df.copy()
    
    # Supprimer les lignes avec des valeurs manquantes
    data.dropna(inplace=True)
    
    # Sélectionner uniquement les colonnes numériques
    numeric_cols = data.select_dtypes(include=['float64', 'int64']).columns.tolist()
    
    # Exclure la variable cible et les colonnes avec trop de valeurs constantes
    features = [col for col in numeric_cols if col != target_var]
    
    # Calculer la corrélation avec la cible
    correlations = data[features].corrwith(data[target_var]).abs().sort_values(ascending=False)
    
    # Sélectionner les top_n caractéristiques
    top_features = correlations.head(top_n).index.tolist()
    
    return top_features


def compute_feature_importance(df: pd.DataFrame, windows: List[int] = [12, 24, 48]) -> Dict[str, pd.DataFrame]:
    """
    Calcule l'importance des caractéristiques pour différents horizons de prédiction
    
    Args:
        df: DataFrame avec les caractéristiques
        windows: Liste des horizons de prédiction (en périodes)
        
    Returns:
        Dictionnaire des importances de caractéristiques par horizon
    """
    importance_by_horizon = {}
    
    for window in windows:
        # Créer la variable cible: direction du prix dans 'window' périodes
        target_var = f'direction_{window}'
        df[target_var] = np.where(df['close'].shift(-window) > df['close'], 1, 0)
        
        # Sélectionner les caractéristiques importantes
        top_features = select_important_features(df, target_var=target_var)
        
        # Calculer l'importance (basée sur la corrélation absolue)
        if top_features and target_var in df.columns:
            importance = df[top_features].corrwith(df[target_var]).abs().sort_values(ascending=False)
            importance_by_horizon[f'horizon_{window}'] = importance
    
    return importance_by_horizon


def normalize_features(df: pd.DataFrame, feature_groups: Dict[str, List[str]] = None) -> pd.DataFrame:
    """
    Normalise les caractéristiques par groupe
    
    Args:
        df: DataFrame avec les caractéristiques
        feature_groups: Dictionnaire des groupes de caractéristiques
        
    Returns:
        DataFrame avec les caractéristiques normalisées
    """
    # Si les groupes ne sont pas fournis, les extraire
    if feature_groups is None:
        feature_groups = extract_feature_groups(df)
    
    # Créer une copie pour éviter de modifier l'original
    normalized_df = df.copy()
    
    # Normaliser chaque groupe séparément
    for group, features in feature_groups.items():
        if not features:
            continue
        
        # Sélectionner uniquement les colonnes numériques
        numeric_features = normalized_df[features].select_dtypes(include=['float64', 'int64']).columns.tolist()
        
        if not numeric_features:
            continue
        
        # Normaliser par groupe (standardisation)
        for col in numeric_features:
            mean = normalized_df[col].mean()
            std = normalized_df[col].std()
            
            if std > 0:
                normalized_df[col] = (normalized_df[col] - mean) / std
            else:
                # Si écart-type nul, remplacer par 0
                normalized_df[col] = 0
    
    return normalized_df