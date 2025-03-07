"""
Module d'analyse technique avec des indicateurs avancés et des fonctions de
détection de patterns pour assister les stratégies de trading
"""
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Union
import datetime
import talib
from scipy import signal

from utils.logger import setup_logger

logger = setup_logger("technical_analysis")

class TechnicalAnalysis:
    """
    Classe utilitaire pour l'analyse technique des marchés financiers,
    fournissant des indicateurs et détecteurs de patterns
    """
    @staticmethod
    def add_indicators(df: pd.DataFrame, indicators: List[str] = None) -> pd.DataFrame:
        """
        Ajoute des indicateurs techniques au DataFrame
        
        Args:
            df: DataFrame avec les données OHLCV
            indicators: Liste des noms d'indicateurs à ajouter
                        (None = tous les indicateurs par défaut)
                        
        Returns:
            DataFrame avec les indicateurs ajoutés
        """
        # Copier le DataFrame pour éviter de modifier l'original
        result = df.copy()
        
        # Vérifier que les colonnes OHLCV existent
        required = ['open', 'high', 'low', 'close', 'volume']
        missing = [col for col in required if col not in df.columns]
        
        if missing:
            logger.error(f"Colonnes manquantes dans le DataFrame: {missing}")
            return df
        
        # Liste des indicateurs disponibles
        available_indicators = {
            'sma': TechnicalAnalysis.add_moving_averages,
            'ema': TechnicalAnalysis.add_moving_averages,
            'rsi': TechnicalAnalysis.add_rsi,
            'macd': TechnicalAnalysis.add_macd,
            'bbands': TechnicalAnalysis.add_bollinger_bands,
            'atr': TechnicalAnalysis.add_atr,
            'stoch': TechnicalAnalysis.add_stochastic,
            'adx': TechnicalAnalysis.add_adx,
            'obv': TechnicalAnalysis.add_volume_indicators,
            'mfi': TechnicalAnalysis.add_volume_indicators,
            'ichimoku': TechnicalAnalysis.add_ichimoku,
            'patterns': TechnicalAnalysis.add_candlestick_patterns,
            'pivots': TechnicalAnalysis.add_pivot_points,
            'trends': TechnicalAnalysis.add_trend_indicators,
            'oscillators': TechnicalAnalysis.add_oscillators
        }
        
        # Indicateurs par défaut
        default_indicators = ['sma', 'ema', 'rsi', 'macd', 'bbands', 'atr', 'stoch', 'adx']
        
        # Déterminer quels indicateurs ajouter
        indicators_to_add = indicators if indicators else default_indicators
        
        # Ajouter chaque indicateur demandé
        for indicator in indicators_to_add:
            if indicator in available_indicators:
                try:
                    result = available_indicators[indicator](result)
                except Exception as e:
                    logger.error(f"Erreur lors de l'ajout de l'indicateur '{indicator}': {str(e)}")
            else:
                logger.warning(f"Indicateur non disponible: {indicator}")
        
        return result
    
    @staticmethod
    def add_moving_averages(df: pd.DataFrame) -> pd.DataFrame:
        """
        Ajoute des moyennes mobiles au DataFrame
        
        Args:
            df: DataFrame avec les données OHLCV
            
        Returns:
            DataFrame avec les moyennes mobiles ajoutées
        """
        result = df.copy()
        
        # Périodes pour les moyennes mobiles
        sma_periods = [5, 10, 20, 50, 100, 200]
        ema_periods = [5, 10, 20, 50, 100, 200]
        
        # Ajouter les SMA
        for period in sma_periods:
            result[f'sma_{period}'] = talib.SMA(result['close'].values, timeperiod=period)
        
        # Ajouter les EMA
        for period in ema_periods:
            result[f'ema_{period}'] = talib.EMA(result['close'].values, timeperiod=period)
        
        # Ajouter les croisements
        if 'sma_20' in result.columns and 'sma_50' in result.columns:
            result['sma_20_50_cross'] = np.where(
                result['sma_20'] > result['sma_50'], 1,
                np.where(result['sma_20'] < result['sma_50'], -1, 0)
            )
        
        if 'ema_20' in result.columns and 'ema_50' in result.columns:
            result['ema_20_50_cross'] = np.where(
                result['ema_20'] > result['ema_50'], 1,
                np.where(result['ema_20'] < result['ema_50'], -1, 0)
            )
        
        # Calculer les distances par rapport aux moyennes mobiles
        for period in [20, 50, 200]:
            if f'sma_{period}' in result.columns:
                result[f'close_to_sma_{period}_pct'] = (result['close'] - result[f'sma_{period}']) / result[f'sma_{period}'] * 100
            
            if f'ema_{period}' in result.columns:
                result[f'close_to_ema_{period}_pct'] = (result['close'] - result[f'ema_{period}']) / result[f'ema_{period}'] * 100
        
        return result
    
    @staticmethod
    def add_rsi(df: pd.DataFrame, periods: List[int] = [7, 14, 21]) -> pd.DataFrame:
        """
        Ajoute l'indicateur RSI au DataFrame
        
        Args:
            df: DataFrame avec les données OHLCV
            periods: Périodes pour le calcul du RSI
            
        Returns:
            DataFrame avec le RSI ajouté
        """
        result = df.copy()
        
        for period in periods:
            result[f'rsi_{period}'] = talib.RSI(result['close'].values, timeperiod=period)
        
        # Zones de surachat/survente
        if 'rsi_14' in result.columns:
            result['rsi_overbought'] = result['rsi_14'] > 70
            result['rsi_oversold'] = result['rsi_14'] < 30
            
            # Divergences RSI-prix
            result['rsi_bullish_div'] = (
                (result['close'] < result['close'].shift(1)) & 
                (result['rsi_14'] > result['rsi_14'].shift(1)) &
                (result['rsi_14'] < 40)
            )
            
            result['rsi_bearish_div'] = (
                (result['close'] > result['close'].shift(1)) & 
                (result['rsi_14'] < result['rsi_14'].shift(1)) &
                (result['rsi_14'] > 60)
            )
        
        return result
    
    @staticmethod
    def add_macd(df: pd.DataFrame, 
               fast_period: int = 12, 
               slow_period: int = 26, 
               signal_period: int = 9) -> pd.DataFrame:
        """
        Ajoute l'indicateur MACD au DataFrame
        
        Args:
            df: DataFrame avec les données OHLCV
            fast_period: Période de l'EMA rapide
            slow_period: Période de l'EMA lente
            signal_period: Période de la ligne de signal
            
        Returns:
            DataFrame avec le MACD ajouté
        """
        result = df.copy()
        
        # Calculer le MACD
        macd, macdsignal, macdhist = talib.MACD(
            result['close'].values, 
            fastperiod=fast_period, 
            slowperiod=slow_period, 
            signalperiod=signal_period
        )
        
        result['macd'] = macd
        result['macd_signal'] = macdsignal
        result['macd_hist'] = macdhist
        
        # Croisements MACD
        result['macd_cross'] = np.where(
            (result['macd'] > result['macd_signal']) & (result['macd'].shift(1) <= result['macd_signal'].shift(1)), 1,
            np.where(
                (result['macd'] < result['macd_signal']) & (result['macd'].shift(1) >= result['macd_signal'].shift(1)), -1, 
                0
            )
        )
        
        # Divergences MACD
        result['macd_bullish_div'] = (
            (result['close'] < result['close'].shift(5)) & 
            (result['macd'] > result['macd'].shift(5)) &
            (result['macd'] < 0)
        )
        
        result['macd_bearish_div'] = (
            (result['close'] > result['close'].shift(5)) & 
            (result['macd'] < result['macd'].shift(5)) &
            (result['macd'] > 0)
        )
        
        return result
    
    @staticmethod
    def add_bollinger_bands(df: pd.DataFrame, 
                         period: int = 20, 
                         num_std_dev: float = 2.0) -> pd.DataFrame:
        """
        Ajoute les bandes de Bollinger au DataFrame
        
        Args:
            df: DataFrame avec les données OHLCV
            period: Période pour le calcul des bandes
            num_std_dev: Nombre d'écarts-types pour les bandes
            
        Returns:
            DataFrame avec les bandes de Bollinger ajoutées
        """
        result = df.copy()
        
        # Calculer les bandes de Bollinger
        upper, middle, lower = talib.BBANDS(
            result['close'].values, 
            timeperiod=period, 
            nbdevup=num_std_dev, 
            nbdevdn=num_std_dev
        )
        
        result['bb_upper'] = upper
        result['bb_middle'] = middle
        result['bb_lower'] = lower
        
        # Calcul de la largeur des bandes et du %B
        result['bb_width'] = (result['bb_upper'] - result['bb_lower']) / result['bb_middle']
        
        # %B indique la position du prix par rapport aux bandes (0 = limite inférieure, 1 = limite supérieure)
        result['bb_percent_b'] = np.where(
            result['bb_upper'] != result['bb_lower'],
            (result['close'] - result['bb_lower']) / (result['bb_upper'] - result['bb_lower']),
            0.5
        )
        
        # Signaux de compression/expansion
        result['bb_squeeze'] = (result['bb_width'] < result['bb_width'].rolling(window=50).quantile(0.2))
        result['bb_expansion'] = (result['bb_width'] > result['bb_width'].rolling(window=50).quantile(0.8))
        
        return result
    
    @staticmethod
    def add_atr(df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
        """
        Ajoute l'indicateur ATR (Average True Range) au DataFrame
        
        Args:
            df: DataFrame avec les données OHLCV
            period: Période pour le calcul de l'ATR
            
        Returns:
            DataFrame avec l'ATR ajouté
        """
        result = df.copy()
        
        result['atr'] = talib.ATR(
            result['high'].values, 
            result['low'].values, 
            result['close'].values, 
            timeperiod=period
        )
        
        # ATR relatif (en pourcentage du prix)
        result['atr_pct'] = result['atr'] / result['close'] * 100
        
        # Changement de volatilité
        result['atr_change'] = result['atr'].pct_change() * 100
        
        return result
    
    @staticmethod
    def add_stochastic(df: pd.DataFrame, 
                     k_period: int = 14, 
                     d_period: int = 3,
                     slowing: int = 3) -> pd.DataFrame:
        """
        Ajoute l'indicateur Stochastique au DataFrame
        
        Args:
            df: DataFrame avec les données OHLCV
            k_period: Période pour le %K
            d_period: Période pour le %D
            slowing: Période de ralentissement
            
        Returns:
            DataFrame avec le Stochastique ajouté
        """
        result = df.copy()
        
        # Calculer le Stochastique
        slowk, slowd = talib.STOCH(
            result['high'].values,
            result['low'].values,
            result['close'].values,
            fastk_period=k_period,
            slowk_period=slowing,
            slowk_matype=0,
            slowd_period=d_period,
            slowd_matype=0
        )
        
        result['stoch_k'] = slowk
        result['stoch_d'] = slowd
        
        # Zones de surachat/survente
        result['stoch_overbought'] = (result['stoch_k'] > 80) & (result['stoch_d'] > 80)
        result['stoch_oversold'] = (result['stoch_k'] < 20) & (result['stoch_d'] < 20)
        
        # Croisements
        result['stoch_cross'] = np.where(
            (result['stoch_k'] > result['stoch_d']) & (result['stoch_k'].shift(1) <= result['stoch_d'].shift(1)), 1,
            np.where(
                (result['stoch_k'] < result['stoch_d']) & (result['stoch_k'].shift(1) >= result['stoch_d'].shift(1)), -1,
                0
            )
        )
        
        # Signaux spécifiques
        result['stoch_bull_cross_oversold'] = (result['stoch_cross'] == 1) & (result['stoch_k'] < 20)
        result['stoch_bear_cross_overbought'] = (result['stoch_cross'] == -1) & (result['stoch_k'] > 80)
        
        return result
    
    @staticmethod
    def add_adx(df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
        """
        Ajoute l'indicateur ADX au DataFrame
        
        Args:
            df: DataFrame avec les données OHLCV
            period: Période pour le calcul de l'ADX
            
        Returns:
            DataFrame avec l'ADX ajouté
        """
        result = df.copy()
        
        # Calculer l'ADX et les lignes DI
        result['adx'] = talib.ADX(
            result['high'].values,
            result['low'].values,
            result['close'].values,
            timeperiod=period
        )
        
        result['di_plus'] = talib.PLUS_DI(
            result['high'].values,
            result['low'].values,
            result['close'].values,
            timeperiod=period
        )
        
        result['di_minus'] = talib.MINUS_DI(
            result['high'].values,
            result['low'].values,
            result['close'].values,
            timeperiod=period
        )
        
        # Forces de tendance
        result['adx_trend_strength'] = np.where(
            result['adx'] < 20, "Weak",
            np.where(
                result['adx'] < 40, "Moderate",
                np.where(
                    result['adx'] < 60, "Strong", "Very Strong"
                )
            )
        )
        
        # Direction de tendance selon l'ADX
        result['adx_trend_direction'] = np.where(
            result['di_plus'] > result['di_minus'], "Bullish",
            np.where(
                result['di_plus'] < result['di_minus'], "Bearish", "Neutral"
            )
        )
        
        # Croisement des lignes DI
        result['di_cross'] = np.where(
            (result['di_plus'] > result['di_minus']) & (result['di_plus'].shift(1) <= result['di_minus'].shift(1)), 1,
            np.where(
                (result['di_plus'] < result['di_minus']) & (result['di_plus'].shift(1) >= result['di_minus'].shift(1)), -1,
                0
            )
        )
        
        return result
    
    @staticmethod
    def add_volume_indicators(df: pd.DataFrame) -> pd.DataFrame:
        """
        Ajoute des indicateurs de volume au DataFrame
        
        Args:
            df: DataFrame avec les données OHLCV
            
        Returns:
            DataFrame avec les indicateurs de volume ajoutés
        """
        result = df.copy()
        
        # On-Balance Volume (OBV)
        result['obv'] = talib.OBV(result['close'].values, result['volume'].values)
        
        # Money Flow Index (MFI)
        result['mfi'] = talib.MFI(
            result['high'].values, 
            result['low'].values, 
            result['close'].values, 
            result['volume'].values, 
            timeperiod=14
        )
        
        # Volume moyen sur différentes périodes
        for period in [5, 10, 20, 50]:
            result[f'volume_sma_{period}'] = result['volume'].rolling(window=period).mean()
        
        # Volume relatif (par rapport à la moyenne)
        if 'volume_sma_20' in result.columns:
            result['volume_ratio'] = result['volume'] / result['volume_sma_20']
        
        # Divergence prix-volume
        result['price_up_volume_up'] = (result['close'] > result['close'].shift(1)) & (result['volume'] > result['volume'].shift(1))
        result['price_up_volume_down'] = (result['close'] > result['close'].shift(1)) & (result['volume'] < result['volume'].shift(1))
        result['price_down_volume_up'] = (result['close'] < result['close'].shift(1)) & (result['volume'] > result['volume'].shift(1))
        result['price_down_volume_down'] = (result['close'] < result['close'].shift(1)) & (result['volume'] < result['volume'].shift(1))
        
        # Volume Force Index
        result['vfi'] = (result['close'] - result['close'].shift(1)) * result['volume']
        result['vfi_ema'] = talib.EMA(result['vfi'].fillna(0).values, timeperiod=13)
        
        return result
    
    @staticmethod
    def add_ichimoku(df: pd.DataFrame,
                   tenkan_period: int = 9,
                   kijun_period: int = 26,
                   senkou_b_period: int = 52,
                   displacement: int = 26) -> pd.DataFrame:
        """
        Ajoute l'indicateur Ichimoku Cloud au DataFrame
        
        Args:
            df: DataFrame avec les données OHLCV
            tenkan_period: Période pour le Tenkan-sen (ligne de conversion)
            kijun_period: Période pour le Kijun-sen (ligne de base)
            senkou_b_period: Période pour le Senkou Span B
            displacement: Décalage pour projeter dans le futur
            
        Returns:
            DataFrame avec l'Ichimoku Cloud ajouté
        """
        result = df.copy()
        
        # Fonction auxiliaire pour calculer la moyenne de haut et bas sur une période
        def donchian(high_vals, low_vals, period):
            high_vals = pd.Series(high_vals)
            low_vals = pd.Series(low_vals)
            return (high_vals.rolling(window=period).max() + low_vals.rolling(window=period).min()) / 2
        
        # Tenkan-sen (ligne de conversion)
        result['ichimoku_tenkan'] = donchian(
            result['high'].values, 
            result['low'].values, 
            tenkan_period
        )
        
        # Kijun-sen (ligne de base)
        result['ichimoku_kijun'] = donchian(
            result['high'].values, 
            result['low'].values, 
            kijun_period
        )
        
        # Chikou Span (ligne de retard)
        result['ichimoku_chikou'] = result['close'].shift(-displacement)
        
        # Senkou Span A (ligne avancée A)
        senkou_span_a = ((result['ichimoku_tenkan'] + result['ichimoku_kijun']) / 2).shift(displacement)
        result['ichimoku_senkou_a'] = senkou_span_a
        
        # Senkou Span B (ligne avancée B)
        senkou_span_b = donchian(
            result['high'].values, 
            result['low'].values, 
            senkou_b_period
        ).shift(displacement)
        result['ichimoku_senkou_b'] = senkou_span_b
        
        # Signal TK Cross (Tenkan/Kijun Cross)
        result['ichimoku_tk_cross'] = np.where(
            (result['ichimoku_tenkan'] > result['ichimoku_kijun']) & 
            (result['ichimoku_tenkan'].shift(1) <= result['ichimoku_kijun'].shift(1)), 1,
            np.where(
                (result['ichimoku_tenkan'] < result['ichimoku_kijun']) & 
                (result['ichimoku_tenkan'].shift(1) >= result['ichimoku_kijun'].shift(1)), -1,
                0
            )
        )
        
        # Kumo breakout (prix traversant le nuage)
        # Calculer le haut et le bas du nuage
        result['ichimoku_kumo_top'] = result[['ichimoku_senkou_a', 'ichimoku_senkou_b']].max(axis=1)
        result['ichimoku_kumo_bottom'] = result[['ichimoku_senkou_a', 'ichimoku_senkou_b']].min(axis=1)
        
        result['ichimoku_kumo_breakout'] = np.where(
            (result['close'] > result['ichimoku_kumo_top']) & 
            (result['close'].shift(1) <= result['ichimoku_kumo_top'].shift(1)), 1,
            np.where(
                (result['close'] < result['ichimoku_kumo_bottom']) & 
                (result['close'].shift(1) >= result['ichimoku_kumo_bottom'].shift(1)), -1,
                0
            )
        )
        
        # État du Kumo (nuage)
        result['ichimoku_kumo_state'] = np.where(
            result['ichimoku_senkou_a'] > result['ichimoku_senkou_b'], "Bullish",
            np.where(
                result['ichimoku_senkou_a'] < result['ichimoku_senkou_b'], "Bearish", "Neutral"
            )
        )
        
        return result
    
    @staticmethod
    def add_candlestick_patterns(df: pd.DataFrame) -> pd.DataFrame:
        """
        Ajoute des patterns de chandeliers au DataFrame
        
        Args:
            df: DataFrame avec les données OHLCV
            
        Returns:
            DataFrame avec les patterns de chandeliers ajoutés
        """
        result = df.copy()
        
        # Patterns haussiers
        result['candle_hammer'] = talib.CDLHAMMER(
            result['open'].values, result['high'].values, 
            result['low'].values, result['close'].values
        )
        
        result['candle_morning_star'] = talib.CDLMORNINGSTAR(
            result['open'].values, result['high'].values, 
            result['low'].values, result['close'].values
        )
        
        result['candle_bullish_engulfing'] = talib.CDLENGULFING(
            result['open'].values, result['high'].values, 
            result['low'].values, result['close'].values
        )
        
        result['candle_piercing'] = talib.CDLPIERCING(
            result['open'].values, result['high'].values, 
            result['low'].values, result['close'].values
        )
        
        # Patterns baissiers
        result['candle_shooting_star'] = talib.CDLSHOOTINGSTAR(
            result['open'].values, result['high'].values, 
            result['low'].values, result['close'].values
        )
        
        result['candle_evening_star'] = talib.CDLEVENINGSTAR(
            result['open'].values, result['high'].values, 
            result['low'].values, result['close'].values
        )
        
        result['candle_bearish_engulfing'] = talib.CDLENGULFING(
            result['open'].values, result['high'].values, 
            result['low'].values, result['close'].values
        ) * -1  # Inverser pour que les valeurs négatives indiquent un pattern baissier
        
        result['candle_dark_cloud_cover'] = talib.CDLDARKCLOUDCOVER(
            result['open'].values, result['high'].values, 
            result['low'].values, result['close'].values
        )
        
        # Patterns de continuation
        result['candle_doji'] = talib.CDLDOJI(
            result['open'].values, result['high'].values, 
            result['low'].values, result['close'].values
        )
        
        result['candle_spinning_top'] = talib.CDLSPINNINGTOP(
            result['open'].values, result['high'].values, 
            result['low'].values, result['close'].values
        )
        
        # Score global de patterns de chandeliers
        # Positif = haussier, négatif = baissier
        bullish_patterns = ['candle_hammer', 'candle_morning_star', 'candle_bullish_engulfing', 'candle_piercing']
        bearish_patterns = ['candle_shooting_star', 'candle_evening_star', 'candle_bearish_engulfing', 'candle_dark_cloud_cover']
        
        result['candle_pattern_score'] = 0
        
        for pattern in bullish_patterns:
            if pattern in result.columns:
                result['candle_pattern_score'] += np.where(result[pattern] > 0, 1, 0)
        
        for pattern in bearish_patterns:
            if pattern in result.columns:
                result['candle_pattern_score'] -= np.where(result[pattern] < 0, 1, 0)
        
        return result
    
    @staticmethod
    def add_pivot_points(df: pd.DataFrame, method: str = 'standard') -> pd.DataFrame:
        """
        Ajoute les points pivots au DataFrame
        
        Args:
            df: DataFrame avec les données OHLCV
            method: Méthode de calcul ('standard', 'fibonacci', 'camarilla', 'woodie')
            
        Returns:
            DataFrame avec les points pivots ajoutés
        """
        result = df.copy()
        
        # Déterminer les valeurs high, low, close précédentes (période précédente)
        # Pour des données journalières, c'est le jour d'avant
        # Pour des timeframes plus courts, on peut grouper par jour
        
        # Créer une fonction qui calcule les pivots basés sur la période précédente
        def calculate_pivots(row, prev_high, prev_low, prev_close):
            # Point pivot central
            if method == 'standard':
                pivot = (prev_high + prev_low + prev_close) / 3
                
                # Supports
                s1 = (2 * pivot) - prev_high
                s2 = pivot - (prev_high - prev_low)
                s3 = pivot - 2 * (prev_high - prev_low)
                
                # Résistances
                r1 = (2 * pivot) - prev_low
                r2 = pivot + (prev_high - prev_low)
                r3 = pivot + 2 * (prev_high - prev_low)
                
            elif method == 'fibonacci':
                pivot = (prev_high + prev_low + prev_close) / 3
                
                # Supports (niveaux Fibonacci)
                s1 = pivot - 0.382 * (prev_high - prev_low)
                s2 = pivot - 0.618 * (prev_high - prev_low)
                s3 = pivot - 1.0 * (prev_high - prev_low)
                
                # Résistances (niveaux Fibonacci)
                r1 = pivot + 0.382 * (prev_high - prev_low)
                r2 = pivot + 0.618 * (prev_high - prev_low)
                r3 = pivot + 1.0 * (prev_high - prev_low)
                
            elif method == 'camarilla':
                pivot = (prev_high + prev_low + prev_close) / 3
                
                # Supports (Camarilla)
                s1 = prev_close - 1.1 / 12.0 * (prev_high - prev_low)
                s2 = prev_close - 1.1 / 6.0 * (prev_high - prev_low)
                s3 = prev_close - 1.1 / 4.0 * (prev_high - prev_low)
                
                # Résistances (Camarilla)
                r1 = prev_close + 1.1 / 12.0 * (prev_high - prev_low)
                r2 = prev_close + 1.1 / 6.0 * (prev_high - prev_low)
                r3 = prev_close + 1.1 / 4.0 * (prev_high - prev_low)
                
            elif method == 'woodie':
                # Woodie utilise l'ouverture de la période actuelle
                current_open = row['open']
                pivot = (prev_high + prev_low + 2 * prev_close) / 4
                
                # Supports
                s1 = (2 * pivot) - prev_high
                s2 = pivot - (prev_high - prev_low)
                s3 = s1 - (prev_high - prev_low)
                
                # Résistances
                r1 = (2 * pivot) - prev_low
                r2 = pivot + (prev_high - prev_low)
                r3 = r1 + (prev_high - prev_low)
                
            else:  # Méthode par défaut
                pivot = (prev_high + prev_low + prev_close) / 3
                
                # Supports
                s1 = (2 * pivot) - prev_high
                s2 = pivot - (prev_high - prev_low)
                s3 = s1 - (prev_high - prev_low)
                
                # Résistances
                r1 = (2 * pivot) - prev_low
                r2 = pivot + (prev_high - prev_low)
                r3 = r1 + (prev_high - prev_low)
            
            return {
                'pivot': pivot,
                'support1': s1,
                'support2': s2,
                'support3': s3,
                'resistance1': r1,
                'resistance2': r2,
                'resistance3': r3
            }
        
        # Appliquer le calcul des pivots (avec une fenêtre glissante)
        result['pivot'] = np.nan
        result['support1'] = np.nan
        result['support2'] = np.nan
        result['support3'] = np.nan
        result['resistance1'] = np.nan
        result['resistance2'] = np.nan
        result['resistance3'] = np.nan
        
        # Pour une approche simple, calculer les pivots en se basant sur la période précédente
        for i in range(1, len(result)):
            prev_high = result.iloc[i-1]['high']
            prev_low = result.iloc[i-1]['low']
            prev_close = result.iloc[i-1]['close']
            
            pivots = calculate_pivots(result.iloc[i], prev_high, prev_low, prev_close)
            
            result.loc[result.index[i], 'pivot'] = pivots['pivot']
            result.loc[result.index[i], 'support1'] = pivots['support1']
            result.loc[result.index[i], 'support2'] = pivots['support2']
            result.loc[result.index[i], 'support3'] = pivots['support3']
            result.loc[result.index[i], 'resistance1'] = pivots['resistance1']
            result.loc[result.index[i], 'resistance2'] = pivots['resistance2']
            result.loc[result.index[i], 'resistance3'] = pivots['resistance3']
        
        # Créer des indicateurs de proximité aux niveaux
        # Par exemple : Le prix est-il proche d'un support ou d'une résistance ?
        result['near_support'] = ((result['close'] - result['support1']).abs() < (result['close'] * 0.005)) | \
                                ((result['close'] - result['support2']).abs() < (result['close'] * 0.005))
        
        result['near_resistance'] = ((result['close'] - result['resistance1']).abs() < (result['close'] * 0.005)) | \
                                   ((result['close'] - result['resistance2']).abs() < (result['close'] * 0.005))
        
        # Ajouter l'information de test des niveaux
        result['pivot_test'] = np.where(
            result['low'] <= result['support1'], -1,
            np.where(
                result['high'] >= result['resistance1'], 1, 0
            )
        )
        
        return result
    
    @staticmethod
    def add_trend_indicators(df: pd.DataFrame) -> pd.DataFrame:
        """
        Ajoute des indicateurs de tendance avancés au DataFrame
        
        Args:
            df: DataFrame avec les données OHLCV
            
        Returns:
            DataFrame avec les indicateurs de tendance ajoutés
        """
        result = df.copy()
        
        # 1. Supertrend Indicator
        # Calculer l'ATR (Average True Range)
        atr_period = 10
        multiplier = 3
        
        if 'atr' not in result.columns:
            result['atr'] = talib.ATR(result['high'].values, result['low'].values, result['close'].values, timeperiod=atr_period)
        
        # Calculer les bandes supérieures et inférieures
        result['supertrend_upper'] = ((result['high'] + result['low']) / 2) + (multiplier * result['atr'])
        result['supertrend_lower'] = ((result['high'] + result['low']) / 2) - (multiplier * result['atr'])
        
        # Initialiser le Supertrend
        result['supertrend'] = 0
        result['supertrend_direction'] = 0
        
        for i in range(1, len(result)):
            # Si le prix de clôture précédent était au-dessus de la bande supérieure
            if result.iloc[i-1]['close'] > result.iloc[i-1]['supertrend_upper']:
                result.loc[result.index[i], 'supertrend'] = result.iloc[i]['supertrend_upper']
                result.loc[result.index[i], 'supertrend_direction'] = 1  # Tendance haussière
            
            # Si le prix de clôture précédent était en-dessous de la bande inférieure
            elif result.iloc[i-1]['close'] < result.iloc[i-1]['supertrend_lower']:
                result.loc[result.index[i], 'supertrend'] = result.iloc[i]['supertrend_lower']
                result.loc[result.index[i], 'supertrend_direction'] = -1  # Tendance baissière
            
            # Sinon, conserver la tendance précédente
            else:
                result.loc[result.index[i], 'supertrend'] = result.iloc[i-1]['supertrend']
                result.loc[result.index[i], 'supertrend_direction'] = result.iloc[i-1]['supertrend_direction']
        
        # 2. Aroon Indicator
        aroon_period = 14
        aroon_up, aroon_down = talib.AROON(result['high'].values, result['low'].values, timeperiod=aroon_period)
        result['aroon_up'] = aroon_up
        result['aroon_down'] = aroon_down
        result['aroon_oscillator'] = aroon_up - aroon_down
        
        # 3. CCI (Commodity Channel Index)
        result['cci'] = talib.CCI(result['high'].values, result['low'].values, result['close'].values, timeperiod=20)
        
        # 4. Détection de tendance basée sur l'ADX
        adx_strength = 25  # Force de tendance minimale
        
        if 'adx' not in result.columns:
            result['adx'] = talib.ADX(result['high'].values, result['low'].values, result['close'].values, timeperiod=14)
        
        if 'di_plus' not in result.columns:
            result['di_plus'] = talib.PLUS_DI(result['high'].values, result['low'].values, result['close'].values, timeperiod=14)
        
        if 'di_minus' not in result.columns:
            result['di_minus'] = talib.MINUS_DI(result['high'].values, result['low'].values, result['close'].values, timeperiod=14)
        
        # Déterminer la tendance basée sur ADX et DI
        result['adx_trend'] = np.where(
            (result['adx'] > adx_strength) & (result['di_plus'] > result['di_minus']), 1,
            np.where(
                (result['adx'] > adx_strength) & (result['di_plus'] < result['di_minus']), -1, 0
            )
        )
        
        # 5. Pente des moyennes mobiles
        # Calculer la pente de la SMA 50 sur les 5 dernières périodes
        if 'sma_50' not in result.columns:
            result['sma_50'] = talib.SMA(result['close'].values, timeperiod=50)
        
        result['sma_50_slope'] = result['sma_50'].diff(5) / 5
        result['sma_50_slope_pct'] = result['sma_50_slope'] / result['sma_50'] * 100
        
        # Calculer la pente relative au prix (forte hausse/baisse)
        result['sma_50_rel_slope'] = np.where(
            result['sma_50_slope_pct'] > 1.0, "Strong Up",
            np.where(
                result['sma_50_slope_pct'] > 0.2, "Up",
                np.where(
                    result['sma_50_slope_pct'] < -1.0, "Strong Down",
                    np.where(
                        result['sma_50_slope_pct'] < -0.2, "Down", "Flat"
                    )
                )
            )
        )
        
        # 6. Score de tendance global
        # Combiner plusieurs indicateurs pour créer un score de tendance
        result['trend_score'] = 0
        
        # Contribution du Supertrend
        result['trend_score'] += result['supertrend_direction']
        
        # Contribution de l'Aroon
        result['trend_score'] += np.where(result['aroon_oscillator'] > 50, 1, np.where(result['aroon_oscillator'] < -50, -1, 0))
        
        # Contribution de l'ADX
        result['trend_score'] += result['adx_trend']
        
        # Contribution du CCI
        result['trend_score'] += np.where(result['cci'] > 100, 1, np.where(result['cci'] < -100, -1, 0))
        
        # Contribution de la pente de la SMA
        result['trend_score'] += np.where(result['sma_50_slope_pct'] > 0.5, 1, np.where(result['sma_50_slope_pct'] < -0.5, -1, 0))
        
        # Catégoriser le score final
        result['trend_strength'] = np.where(
            result['trend_score'] >= 4, "Very Strong Up",
            np.where(
                result['trend_score'] >= 2, "Strong Up",
                np.where(
                    result['trend_score'] >= 1, "Weak Up",
                    np.where(
                        result['trend_score'] <= -4, "Very Strong Down",
                        np.where(
                            result['trend_score'] <= -2, "Strong Down",
                            np.where(
                                result['trend_score'] <= -1, "Weak Down", "No Trend"
                            )
                        )
                    )
                )
            )
        )
        
        return result
    
    @staticmethod
    def add_oscillators(df: pd.DataFrame) -> pd.DataFrame:
        """
        Ajoute des oscillateurs avancés au DataFrame
        
        Args:
            df: DataFrame avec les données OHLCV
            
        Returns:
            DataFrame avec les oscillateurs ajoutés
        """
        result = df.copy()
        
        # 1. Awesome Oscillator
        # (SMA(HIGH+LOW, 5) - SMA(HIGH+LOW, 34)) / 2
        median_price = (result['high'] + result['low']) / 2
        ao_fast = median_price.rolling(window=5).mean()
        ao_slow = median_price.rolling(window=34).mean()
        result['awesome_oscillator'] = ao_fast - ao_slow
        
        # AO Signaux
        result['ao_crossover'] = np.where(
            (result['awesome_oscillator'] > 0) & (result['awesome_oscillator'].shift(1) <= 0), 1,
            np.where(
                (result['awesome_oscillator'] < 0) & (result['awesome_oscillator'].shift(1) >= 0), -1, 0
            )
        )
        
        # 2. Williams %R
        result['williams_r'] = talib.WILLR(
            result['high'].values,
            result['low'].values,
            result['close'].values,
            timeperiod=14
        )
        
        # Williams %R - surachat/survente
        result['williams_overbought'] = result['williams_r'] > -20
        result['williams_oversold'] = result['williams_r'] < -80
        
        # 3. Ultimate Oscillator
        result['ultimate_oscillator'] = talib.ULTOSC(
            result['high'].values,
            result['low'].values,
            result['close'].values,
            timeperiod1=7,
            timeperiod2=14,
            timeperiod3=28
        )
        
        # UO Signaux
        result['uo_overbought'] = result['ultimate_oscillator'] > 70
        result['uo_oversold'] = result['ultimate_oscillator'] < 30
        
        # 4. PPO (Percentage Price Oscillator)
        # Similaire au MACD mais en pourcentage
        result['ppo'], result['ppo_signal'], result['ppo_hist'] = talib.PPO(
            result['close'].values,
            fastperiod=12,
            slowperiod=26,
            matype=0
        )
        
        # PPO Crossover
        result['ppo_crossover'] = np.where(
            (result['ppo'] > result['ppo_signal']) & (result['ppo'].shift(1) <= result['ppo_signal'].shift(1)), 1,
            np.where(
                (result['ppo'] < result['ppo_signal']) & (result['ppo'].shift(1) >= result['ppo_signal'].shift(1)), -1, 0
            )
        )
        
        # 5. CMO (Chande Momentum Oscillator)
        result['cmo'] = talib.CMO(result['close'].values, timeperiod=14)
        
        # CMO Signaux
        result['cmo_overbought'] = result['cmo'] > 50
        result['cmo_oversold'] = result['cmo'] < -50
        
        # 6. TSI (True Strength Index)
        # Calcul manuel car non disponible dans talib
        def calc_tsi(close, r=25, s=13, u=7):
            m = pd.Series(close.diff())  # Momentum
            
            # Double EMA lissée du momentum
            m1 = m.ewm(span=r, adjust=False).mean()  # Premier smooth
            m2 = m1.ewm(span=s, adjust=False).mean()  # Deuxième smooth
            
            # Double EMA lissée du momentum absolu
            a1 = m.abs().ewm(span=r, adjust=False).mean()
            a2 = a1.ewm(span=s, adjust=False).mean()
            
            # TSI = (m2 / a2) * 100
            tsi = 100 * m2 / a2
            
            # Signal line
            signal = tsi.ewm(span=u, adjust=False).mean()
            
            return tsi, signal
        
        result['tsi'], result['tsi_signal'] = calc_tsi(result['close'])
        
        # TSI Signaux
        result['tsi_crossover'] = np.where(
            (result['tsi'] > result['tsi_signal']) & (result['tsi'].shift(1) <= result['tsi_signal'].shift(1)), 1,
            np.where(
                (result['tsi'] < result['tsi_signal']) & (result['tsi'].shift(1) >= result['tsi_signal'].shift(1)), -1, 0
            )
        )
        
        # 7. Score d'oscillateur composite
        # Combine plusieurs oscillateurs pour un signal plus robuste
        result['oscillator_score'] = 0
        
        # RSI contribution (si disponible)
        if 'rsi_14' in result.columns:
            result['oscillator_score'] += np.where(
                result['rsi_14'] > 70, -1,
                np.where(
                    result['rsi_14'] < 30, 1, 0
                )
            )
        
        # Williams %R contribution
        result['oscillator_score'] += np.where(
            result['williams_r'] > -20, -1,
            np.where(
                result['williams_r'] < -80, 1, 0
            )
        )
        
        # Ultimate Oscillator contribution
        result['oscillator_score'] += np.where(
            result['ultimate_oscillator'] > 70, -1,
            np.where(
                result['ultimate_oscillator'] < 30, 1, 0
            )
        )
        
        # CMO contribution
        result['oscillator_score'] += np.where(
            result['cmo'] > 50, -1,
            np.where(
                result['cmo'] < -50, 1, 0
            )
        )
        
        # Signaux de l'oscillateur composite
        result['oscillator_buy_signal'] = result['oscillator_score'] >= 2
        result['oscillator_sell_signal'] = result['oscillator_score'] <= -2
        
        return result

    @staticmethod
    def detect_support_resistance(df: pd.DataFrame, window_size: int = 20, threshold: float = 0.01) -> Dict:
        """
        Détecte les niveaux de support et résistance
        
        Args:
            df: DataFrame avec les données OHLCV
            window_size: Taille de la fenêtre pour la recherche
            threshold: Seuil de distance relative
            
        Returns:
            Dictionnaire avec les niveaux de support et résistance
        """
        data = df.copy()
        
        # 1. Détecter les hauts et bas locaux
        data['is_high'] = np.zeros(len(data))
        data['is_low'] = np.zeros(len(data))
        
        for i in range(window_size, len(data) - window_size):
            # Un maximum local est trouvé si le prix est le plus haut dans la fenêtre
            if data['high'].iloc[i] == data['high'].iloc[i-window_size:i+window_size+1].max():
                data.loc[data.index[i], 'is_high'] = 1
            
            # Un minimum local est trouvé si le prix est le plus bas dans la fenêtre
            if data['low'].iloc[i] == data['low'].iloc[i-window_size:i+window_size+1].min():
                data.loc[data.index[i], 'is_low'] = 1
        
        # 2. Extraire les niveaux de prix correspondants aux hauts et bas locaux
        highs = data[data['is_high'] == 1]['high'].values
        lows = data[data['is_low'] == 1]['low'].values
        
        # 3. Regrouper les niveaux proches (clustering)
        def cluster_levels(levels, threshold):
            # Trier les niveaux
            sorted_levels = np.sort(levels)
            
            # Créer des clusters
            clusters = []
            current_cluster = [sorted_levels[0]]
            
            for i in range(1, len(sorted_levels)):
                # Si ce niveau est proche du dernier niveau dans le cluster actuel
                if sorted_levels[i] / current_cluster[-1] - 1 < threshold:
                    # Ajouter au cluster actuel
                    current_cluster.append(sorted_levels[i])
                else:
                    # Finaliser le cluster actuel et en commencer un nouveau
                    clusters.append(current_cluster)
                    current_cluster = [sorted_levels[i]]
            
            # Ajouter le dernier cluster
            if current_cluster:
                clusters.append(current_cluster)
            
            # Calculer la moyenne de chaque cluster
            return [np.mean(cluster) for cluster in clusters]
        
        resistance_levels = cluster_levels(highs, threshold)
        support_levels = cluster_levels(lows, threshold)
        
        # 4. Évaluer la force des niveaux (nombre de tests)
        resistance_strength = {}
        support_strength = {}
        
        for level in resistance_levels:
            # Compter le nombre de fois où le prix s'approche du niveau de résistance
            count = np.sum(np.abs(data['high'] - level) / level < threshold)
            resistance_strength[level] = count
        
        for level in support_levels:
            # Compter le nombre de fois où le prix s'approche du niveau de support
            count = np.sum(np.abs(data['low'] - level) / level < threshold)
            support_strength[level] = count
        
        # 5. Trier les niveaux par force décroissante
        resistance_levels = sorted(resistance_strength.items(), key=lambda x: x[1], reverse=True)
        support_levels = sorted(support_strength.items(), key=lambda x: x[1], reverse=True)
        
        # 6. Détecter les niveaux actuels les plus pertinents
        current_price = data['close'].iloc[-1]
        
        # Résistances au-dessus du prix actuel
        resistances_above = [(level, strength) for level, strength in resistance_levels if level > current_price]
        
        # Supports en-dessous du prix actuel
        supports_below = [(level, strength) for level, strength in support_levels if level < current_price]
        
        # 7. Préparer les résultats
        return {
            "support_levels": support_levels,
            "resistance_levels": resistance_levels,
            "nearest_support": supports_below[0][0] if supports_below else None,
            "nearest_resistance": resistances_above[0][0] if resistances_above else None,
            "current_price": current_price
        }

    @staticmethod
    def detect_breakouts(df: pd.DataFrame, lookback_period: int = 20) -> Dict:
        """
        Détecte les breakouts de supports, résistances et patterns
        
        Args:
            df: DataFrame avec les données OHLCV
            lookback_period: Période de lookback pour les niveaux
            
        Returns:
            Dictionnaire avec les informations de breakout
        """
        data = df.copy()
        
        # Résultats
        breakouts = {
            "price_breakout": None,
            "volume_breakout": False,
            "volatility_breakout": False,
            "pattern_breakout": None,
            "momentum_confirmation": False,
            "score": 0,  # Score global de breakout
            "signals": []  # Liste de signaux détectés
        }
        
        # Vérifier la taille des données
        if len(data) < lookback_period + 5:
            breakouts["error"] = "Insufficient data points"
            return breakouts
        
        # Prix actuel et antérieur
        current_price = data['close'].iloc[-1]
        prior_close = data['close'].iloc[-2]
        
        # 1. Détection de breakout de prix (résistance ou support)
        # Calculer la résistance récente
        recent_high = data['high'].iloc[-lookback_period-1:-1].max()
        recent_low = data['low'].iloc[-lookback_period-1:-1].min()
        
        # Range trading récent
        recent_range = recent_high - recent_low
        
        # Vérifier le breakout de prix
        if current_price > recent_high * 1.01:  # 1% au-dessus du plus haut récent
            breakouts["price_breakout"] = "bullish"
            breakouts["score"] += 2
            breakouts["signals"].append("Bullish price breakout")
        elif current_price < recent_low * 0.99:  # 1% en-dessous du plus bas récent
            breakouts["price_breakout"] = "bearish"
            breakouts["score"] += 2
            breakouts["signals"].append("Bearish price breakout")
        
        # 2. Volume Breakout
        # Calculer le volume moyen récent et vérifier si le volume actuel est bien supérieur
        avg_volume = data['volume'].iloc[-lookback_period-1:-1].mean()
        current_volume = data['volume'].iloc[-1]
        
        if current_volume > 2 * avg_volume:  # Volume double de la moyenne
            breakouts["volume_breakout"] = True
            breakouts["score"] += 1
            breakouts["signals"].append("Volume breakout")
        
        # 3. Volatility Breakout
        # Calculer l'ATR pour la volatilité
        atr = talib.ATR(data['high'].values, data['low'].values, data['close'].values, timeperiod=14)
        avg_atr = atr.iloc[-lookback_period-1:-1].mean()
        current_range = data['high'].iloc[-1] - data['low'].iloc[-1]
        
        if current_range > 2 * avg_atr:
            breakouts["volatility_breakout"] = True
            breakouts["score"] += 1
            breakouts["signals"].append("Volatility breakout")
        
        # 4. Pattern Breakout
        # Vérifier les patterns de chandeliers récents
        try:
            # Utiliser les motifs de chandeliers de TA-Lib
            doji = talib.CDLDOJI(data['open'].values, data['high'].values, data['low'].values, data['close'].values)
            hammer = talib.CDLHAMMER(data['open'].values, data['high'].values, data['low'].values, data['close'].values)
            engulfing = talib.CDLENGULFING(data['open'].values, data['high'].values, data['low'].values, data['close'].values)
            
            # Vérifier si un modèle bullish apparaît dans les 3 dernières bougies
            recent_bullish = (hammer[-3:] > 0).any() or (engulfing[-3:] > 0).any()
            
            # Vérifier si un modèle bearish apparaît dans les 3 dernières bougies
            recent_bearish = (engulfing[-3:] < 0).any()
            
            if breakouts["price_breakout"] == "bullish" and recent_bullish:
                breakouts["pattern_breakout"] = "bullish_confirmed"
                breakouts["score"] += 2
                breakouts["signals"].append("Bullish pattern confirmation")
            elif breakouts["price_breakout"] == "bearish" and recent_bearish:
                breakouts["pattern_breakout"] = "bearish_confirmed"
                breakouts["score"] += 2
                breakouts["signals"].append("Bearish pattern confirmation")
            elif recent_bullish:
                breakouts["pattern_breakout"] = "bullish"
                breakouts["score"] += 1
                breakouts["signals"].append("Bullish pattern detected")
            elif recent_bearish:
                breakouts["pattern_breakout"] = "bearish"
                breakouts["score"] += 1
                breakouts["signals"].append("Bearish pattern detected")
        except Exception as e:
            pass  # Ignorer les erreurs dans la détection de motifs
        
        # 5. Momentum confirmation
        # Vérifier si le RSI confirme le mouvement
        try:
            rsi = talib.RSI(data['close'].values, timeperiod=14)
            current_rsi = rsi[-1]
            prior_rsi = rsi[-2]
            
            # Pour un breakout haussier, RSI devrait être en hausse
            if breakouts["price_breakout"] == "bullish" and current_rsi > prior_rsi and current_rsi > 50:
                breakouts["momentum_confirmation"] = True
                breakouts["score"] += 1
                breakouts["signals"].append("RSI confirms bullish momentum")
            # Pour un breakout baissier, RSI devrait être en baisse
            elif breakouts["price_breakout"] == "bearish" and current_rsi < prior_rsi and current_rsi < 50:
                breakouts["momentum_confirmation"] = True
                breakouts["score"] += 1
                breakouts["signals"].append("RSI confirms bearish momentum")
        except Exception as e:
            pass  # Ignorer les erreurs dans le calcul du RSI
        
        # 6. Interprétation globale
        if breakouts["score"] >= 4:
            if breakouts["price_breakout"] == "bullish":
                breakouts["interpretation"] = "Strong bullish breakout"
                breakouts["strength"] = "high"
            elif breakouts["price_breakout"] == "bearish":
                breakouts["interpretation"] = "Strong bearish breakdown"
                breakouts["strength"] = "high"
        elif breakouts["score"] >= 2:
            if breakouts["price_breakout"] == "bullish":
                breakouts["interpretation"] = "Potential bullish breakout"
                breakouts["strength"] = "medium"
            elif breakouts["price_breakout"] == "bearish":
                breakouts["interpretation"] = "Potential bearish breakdown"
                breakouts["strength"] = "medium"
        else:
            breakouts["interpretation"] = "No significant breakout detected"
            breakouts["strength"] = "low"
        
        return breakouts

    @staticmethod
    def detect_divergences(df: pd.DataFrame, lookahead: int = 5) -> Dict:
        """
        Détecte les divergences entre le prix et les oscillateurs (RSI, MACD)
        
        Args:
            df: DataFrame avec les données OHLCV
            lookahead: Nombre de périodes à inclure pour la confirmation
            
        Returns:
            Dictionnaire avec les informations sur les divergences
        """
        data = df.copy()
        result = {
            "rsi_bullish_div": False,
            "rsi_bearish_div": False,
            "macd_bullish_div": False,
            "macd_bearish_div": False,
            "divergence_score": 0,
            "signals": []
        }
        
        # Minimum de données requises
        min_periods = 50
        if len(data) < min_periods:
            result["error"] = f"Insufficient data, need at least {min_periods} periods"
            return result
        
        # Calculer les indicateurs nécessaires s'ils n'existent pas déjà
        if 'rsi_14' not in data.columns:
            data['rsi_14'] = talib.RSI(data['close'].values, timeperiod=14)
        
        if 'macd' not in data.columns or 'macd_signal' not in data.columns:
            data['macd'], data['macd_signal'], _ = talib.MACD(
                data['close'].values,
                fastperiod=12,
                slowperiod=26,
                signalperiod=9
            )
        
        # Fonction pour détecter les sommets et creux
        def detect_peaks(series, window_size=5):
            peaks = []
            troughs = []
            
            for i in range(window_size, len(series) - window_size):
                if all(series[i] > series[i-j] for j in range(1, window_size+1)) and \
                   all(series[i] > series[i+j] for j in range(1, window_size+1)):
                    peaks.append(i)
                
                if all(series[i] < series[i-j] for j in range(1, window_size+1)) and \
                   all(series[i] < series[i+j] for j in range(1, window_size+1)):
                    troughs.append(i)
            
            return peaks, troughs
        
        # Détecter les sommets et creux du prix
        price_peaks, price_troughs = detect_peaks(data['close'].values)
        
        # Détecter les sommets et creux du RSI
        rsi_peaks, rsi_troughs = detect_peaks(data['rsi_14'].values)
        
        # Détecter les sommets et creux du MACD
        macd_peaks, macd_troughs = detect_peaks(data['macd'].values)
        
        # Vérifier les divergences RSI
        # Divergence haussière: prix fait des creux plus bas mais RSI fait des creux plus hauts
        for i, price_idx in enumerate(price_troughs[:-1]):
            for j, rsi_idx in enumerate(rsi_troughs):
                # Les indices doivent être proches (dans une fenêtre)
                if abs(price_idx - rsi_idx) <= lookahead:
                    # Chercher le prochain creux du prix et du RSI
                    if i + 1 < len(price_troughs) and j + 1 < len(rsi_troughs):
                        next_price_idx = price_troughs[i + 1]
                        next_rsi_idx = rsi_troughs[j + 1]
                        
                        # Divergence haussière: prix plus bas mais RSI plus haut
                        if (data['close'][next_price_idx] < data['close'][price_idx] and
                            data['rsi_14'][next_rsi_idx] > data['rsi_14'][rsi_idx]):
                            result["rsi_bullish_div"] = True
                            result["divergence_score"] += 1
                            result["signals"].append("RSI bullish divergence detected")
                            break
        
        # Divergence baissière: prix fait des sommets plus hauts mais RSI fait des sommets plus bas
        for i, price_idx in enumerate(price_peaks[:-1]):
            for j, rsi_idx in enumerate(rsi_peaks):
                # Les indices doivent être proches (dans une fenêtre)
                if abs(price_idx - rsi_idx) <= lookahead:
                    # Chercher le prochain sommet du prix et du RSI
                    if i + 1 < len(price_peaks) and j + 1 < len(rsi_peaks):
                        next_price_idx = price_peaks[i + 1]
                        next_rsi_idx = rsi_peaks[j + 1]
                        
                        # Divergence baissière: prix plus haut mais RSI plus bas
                        if (data['close'][next_price_idx] > data['close'][price_idx] and
                            data['rsi_14'][next_rsi_idx] < data['rsi_14'][rsi_idx]):
                            result["rsi_bearish_div"] = True
                            result["divergence_score"] += 1
                            result["signals"].append("RSI bearish divergence detected")
                            break
        
        # Vérifier les divergences MACD (similaire au RSI)
        # Divergence haussière: prix fait des creux plus bas mais MACD fait des creux plus hauts
        for i, price_idx in enumerate(price_troughs[:-1]):
            for j, macd_idx in enumerate(macd_troughs):
                if abs(price_idx - macd_idx) <= lookahead:
                    if i + 1 < len(price_troughs) and j + 1 < len(macd_troughs):
                        next_price_idx = price_troughs[i + 1]
                        next_macd_idx = macd_troughs[j + 1]
                        
                        if (data['close'][next_price_idx] < data['close'][price_idx] and
                            data['macd'][next_macd_idx] > data['macd'][macd_idx]):
                            result["macd_bullish_div"] = True
                            result["divergence_score"] += 1
                            result["signals"].append("MACD bullish divergence detected")
                            break
        
        # Divergence baissière: prix fait des sommets plus hauts mais MACD fait des sommets plus bas
        for i, price_idx in enumerate(price_peaks[:-1]):
            for j, macd_idx in enumerate(macd_peaks):
                if abs(price_idx - macd_idx) <= lookahead:
                    if i + 1 < len(price_peaks) and j + 1 < len(macd_peaks):
                        next_price_idx = price_peaks[i + 1]
                        next_macd_idx = macd_peaks[j + 1]
                        
                        if (data['close'][next_price_idx] > data['close'][price_idx] and
                            data['macd'][next_macd_idx] < data['macd'][macd_idx]):
                            result["macd_bearish_div"] = True
                            result["divergence_score"] += 1
                            result["signals"].append("MACD bearish divergence detected")
                            break
        
        # Interprétation globale
        if result["divergence_score"] >= 2:
            if result["rsi_bullish_div"] or result["macd_bullish_div"]:
                result["interpretation"] = "Strong bullish divergence"
                result["direction"] = "bullish"
            else:
                result["interpretation"] = "Strong bearish divergence"
                result["direction"] = "bearish"
        elif result["divergence_score"] == 1:
            if result["rsi_bullish_div"] or result["macd_bullish_div"]:
                result["interpretation"] = "Potential bullish divergence"
                result["direction"] = "bullish"
            else:
                result["interpretation"] = "Potential bearish divergence"
                result["direction"] = "bearish"
        else:
            result["interpretation"] = "No significant divergence detected"
            result["direction"] = "neutral"
        
        return result

    @staticmethod
    def get_market_structure(df: pd.DataFrame, window: int = 10) -> Dict:
        """
        Analyse la structure du marché pour identifier les tendances, ranges, et points de retournement
        
        Args:
            df: DataFrame avec les données OHLCV
            window: Fenêtre pour les calculs de structure
            
        Returns:
            Dictionnaire avec les informations sur la structure du marché
        """
        data = df.copy()
        result = {
            "market_structure": "unknown",
            "strength": 0,
            "confidence": 0.0,
            "patterns": [],
            "key_levels": {}
        }
        
        # Minimum de données requises
        if len(data) < 30:
            result["error"] = "Insufficient data, need at least 30 periods"
            return result
        
        try:
            # 1. Détecter les supports et résistances
            sr_levels = TechnicalAnalysis.detect_support_resistance(data)
            result["key_levels"] = {
                "support": sr_levels.get("nearest_support"),
                "resistance": sr_levels.get("nearest_resistance"),
                "all_supports": [level for level, _ in sr_levels.get("support_levels", [])][:3],
                "all_resistances": [level for level, _ in sr_levels.get("resistance_levels", [])][:3]
            }
            
            # 2. Identifier les sommets/creux récents
            recent = data.iloc[-window:]
            is_higher_high = recent['high'].iloc[-1] > recent['high'].iloc[:-1].max()
            is_lower_low = recent['low'].iloc[-1] < recent['low'].iloc[:-1].min()
            
            # 3. Analyser les indicateurs de tendance
            if 'adx' not in data.columns:
                data['adx'] = talib.ADX(data['high'].values, data['low'].values, data['close'].values, timeperiod=14)
            
            if 'sma_50' not in data.columns:
                data['sma_50'] = talib.SMA(data['close'].values, timeperiod=50)
            
            if 'sma_200' not in data.columns:
                data['sma_200'] = talib.SMA(data['close'].values, timeperiod=200)
            
            # Analyser les moyennes mobiles pour déterminer la tendance
            price = data['close'].iloc[-1]
            sma50 = data['sma_50'].iloc[-1]
            sma200 = data['sma_200'].iloc[-1]
            
            above_sma50 = price > sma50
            above_sma200 = price > sma200
            
            # Force de tendance avec ADX
            adx_value = data['adx'].iloc[-1]
            strong_trend = adx_value > 25
            
            # 4. Analyser la volatilité
            if 'atr' not in data.columns:
                data['atr'] = talib.ATR(data['high'].values, data['low'].values, data['close'].values, timeperiod=14)
            
            atr = data['atr'].iloc[-1]
            atr_percent = (atr / price) * 100  # ATR en pourcentage du prix
            
            is_high_volatility = atr_percent > 3.0  # Seuil arbitraire, à ajuster selon le marché
            
            # 5. Détecter les patterns de chandeliers
            candle_patterns = []
            
            # Hammer ou Shooting Star
            hammer = talib.CDLHAMMER(data['open'].values, data['high'].values, data['low'].values, data['close'].values)
            if hammer[-3:].any() != 0:
                candle_patterns.append("hammer")
            
            shooting_star = talib.CDLSHOOTINGSTAR(data['open'].values, data['high'].values, data['low'].values, data['close'].values)
            if shooting_star[-3:].any() != 0:
                candle_patterns.append("shooting_star")
            
            engulfing = talib.CDLENGULFING(data['open'].values, data['high'].values, data['low'].values, data['close'].values)
            if engulfing[-3:].any() != 0:
                candle_patterns.append("engulfing")
            
            doji = talib.CDLDOJI(data['open'].values, data['high'].values, data['low'].values, data['close'].values)
            if doji[-3:].any() != 0:
                candle_patterns.append("doji")
            
            # 6. Déterminer la structure du marché
            if strong_trend:
                if above_sma50 and above_sma200 and is_higher_high:
                    result["market_structure"] = "strong_uptrend"
                    result["strength"] = int(adx_value)
                    result["confidence"] = min(1.0, adx_value / 50)
                    result["patterns"].extend(candle_patterns)
                elif not above_sma50 and not above_sma200 and is_lower_low:
                    result["market_structure"] = "strong_downtrend"
                    result["strength"] = int(adx_value)
                    result["confidence"] = min(1.0, adx_value / 50)
                    result["patterns"].extend(candle_patterns)
                else:
                    # Tendance moins claire
                    if above_sma50:
                        result["market_structure"] = "weak_uptrend"
                        result["strength"] = int(adx_value * 0.8)
                        result["confidence"] = min(1.0, adx_value / 60)
                    else:
                        result["market_structure"] = "weak_downtrend"
                        result["strength"] = int(adx_value * 0.8)
                        result["confidence"] = min(1.0, adx_value / 60)
            else:
                # Faible tendance, probablement consolidation ou range
                high_low_diff = data['high'].iloc[-window:].max() - data['low'].iloc[-window:].min()
                price_range = high_low_diff / price * 100  # En pourcentage
                
                if price_range < 5.0:  # Seuil arbitraire pour une consolidation étroite
                    result["market_structure"] = "tight_consolidation"
                    result["strength"] = int(10 - price_range)  # Plus le range est petit, plus la force est grande
                    result["patterns"].extend(candle_patterns)
                    
                    # Si doji présent dans une consolidation = indécision
                    if "doji" in candle_patterns:
                        result["patterns"].append("indecision")
                else:
                    result["market_structure"] = "range_bound"
                    result["strength"] = int(5 - min(5, adx_value/5))  # Moins l'ADX est élevé, plus le range est fort
                    result["patterns"].extend(candle_patterns)
            
            # 7. Détecter les patterns de retournement potentiels
            if "hammer" in candle_patterns and result["market_structure"].endswith("downtrend"):
                result["patterns"].append("potential_reversal_up")
            
            if "shooting_star" in candle_patterns and result["market_structure"].endswith("uptrend"):
                result["patterns"].append("potential_reversal_down")
            
            # 8. Analyser les volumes pour confirmation
            avg_volume = data['volume'].iloc[-window*2:-1].mean()
            current_volume = data['volume'].iloc[-1]
            volume_ratio = current_volume / avg_volume
            
            result["volume_analysis"] = {
                "ratio": float(volume_ratio),
                "is_high": volume_ratio > 1.5,
                "is_low": volume_ratio < 0.5
            }
            
            # Volume confirme la tendance ?
            if result["market_structure"].endswith("uptrend") and volume_ratio > 1.5:
                result["patterns"].append("volume_confirms_uptrend")
                result["confidence"] = min(1.0, result["confidence"] + 0.1)
            
            if result["market_structure"].endswith("downtrend") and volume_ratio > 1.5:
                result["patterns"].append("volume_confirms_downtrend")
                result["confidence"] = min(1.0, result["confidence"] + 0.1)
        
        except Exception as e:
            result["error"] = f"Error in market structure analysis: {str(e)}"
        
        return result

    @staticmethod
    def get_technical_summary(df: pd.DataFrame, timeframe: str = 'any') -> Dict:
        """
        Génère un résumé technique complet pour une analyse rapide du marché
        
        Args:
            df: DataFrame avec les données OHLCV
            timeframe: Timeframe des données (pour l'interprétation)
            
        Returns:
            Dictionnaire avec le résumé technique
        """
        data = df.copy()
        
        # Vérifier que nous avons suffisamment de données
        if len(data) < 200:
            return {
                "error": "Insufficient data for complete analysis, need at least 200 periods",
                "available_data_points": len(data)
            }
        
        # Résultat global
        summary = {
            "trend": {
                "direction": "unknown",
                "strength": 0,
                "description": ""
            },
            "momentum": {
                "state": "unknown",
                "strength": 0,
                "description": ""
            },
            "volatility": {
                "level": "unknown",
                "description": ""
            },
            "support_resistance": {
                "nearest_support": None,
                "nearest_resistance": None,
                "distance_to_support": None,
                "distance_to_resistance": None
            },
            "signals": [],
            "patterns": [],
            "overall_bias": "neutral",
            "timeframe": timeframe,
            "timestamp": datetime.now().isoformat()
        }
        
        try:
            # 1. Ajouter les indicateurs nécessaires
            # Tendance - ADX, moyennes mobiles
            result = TechnicalAnalysis.add_adx(data)
            result = TechnicalAnalysis.add_moving_averages(result)
            
            # Momentum - RSI, MACD, Stochastique
            result = TechnicalAnalysis.add_rsi(result)
            result = TechnicalAnalysis.add_macd(result)
            result = TechnicalAnalysis.add_stochastic(result)
            
            # Volatilité - ATR, Bandes de Bollinger
            result = TechnicalAnalysis.add_atr(result)
            result = TechnicalAnalysis.add_bollinger_bands(result)
            
            # 2. Analyser la tendance
            adx = result['adx'].iloc[-1]
            di_plus = result['di_plus'].iloc[-1]
            di_minus = result['di_minus'].iloc[-1]
            
            # Direction de tendance basée sur plusieurs indicateurs
            trend_indicators = []
            
            # ADX Direction
            if adx > 20:
                if di_plus > di_minus:
                    trend_indicators.append(1)  # Haussière
                else:
                    trend_indicators.append(-1)  # Baissière
            
            # Moyennes mobiles (prix au-dessus ou en-dessous)
            price = result['close'].iloc[-1]
            
            for period in [50, 200]:
                ma_col = f'sma_{period}'
                if ma_col in result.columns:
                    ma_value = result[ma_col].iloc[-1]
                    if not np.isnan(ma_value):
                        if price > ma_value:
                            trend_indicators.append(1)  # Haussière
                        else:
                            trend_indicators.append(-1)  # Baissière
            
            # Supertrend (si disponible)
            if 'supertrend_direction' in result.columns:
                trend_indicators.append(result['supertrend_direction'].iloc[-1])
            
            # Compter les votes pour chaque direction
            bullish_votes = sum(1 for x in trend_indicators if x > 0)
            bearish_votes = sum(1 for x in trend_indicators if x < 0)
            
            # Déterminer la direction et la force de la tendance
            if bullish_votes > bearish_votes:
                trend_direction = "bullish"
                trend_strength = int(min(100, (bullish_votes / len(trend_indicators)) * 100))
            elif bearish_votes > bullish_votes:
                trend_direction = "bearish"
                trend_strength = int(min(100, (bearish_votes / len(trend_indicators)) * 100))
            else:
                trend_direction = "neutral"
                trend_strength = 0
            
            # Force de la tendance basée sur ADX
            if adx < 20:
                trend_strength = max(0, trend_strength - 50)
                trend_description = "Weak or no trend"
            elif adx < 30:
                trend_description = "Moderate trend"
            else:
                trend_strength = min(100, trend_strength + 20)
                trend_description = "Strong trend"
            
            summary["trend"]["direction"] = trend_direction
            summary["trend"]["strength"] = trend_strength
            summary["trend"]["description"] = trend_description
            
            # 3. Analyser le momentum
            # RSI
            rsi = result['rsi_14'].iloc[-1]
            
            # MACD
            macd = result['macd'].iloc[-1]
            macd_signal = result['macd_signal'].iloc[-1]
            macd_hist = result['macd_hist'].iloc[-1]
            
            # Stochastique
            stoch_k = result['stoch_k'].iloc[-1]
            stoch_d = result['stoch_d'].iloc[-1]
            
            # Compter les signaux de momentum
            momentum_indicators = []
            
            # RSI
            if rsi > 70:
                momentum_indicators.append(-1)  # Survente (signal baissier)
                summary["signals"].append("RSI overbought")
            elif rsi < 30:
                momentum_indicators.append(1)  # Surachat (signal haussier)
                summary["signals"].append("RSI oversold")
            elif rsi > 50:
                momentum_indicators.append(0.5)  # Légèrement haussier
            elif rsi < 50:
                momentum_indicators.append(-0.5)  # Légèrement baissier
            
            # MACD
            if macd > macd_signal:
                momentum_indicators.append(1)  # Haussier
            else:
                momentum_indicators.append(-1)  # Baissier
            
            # MACD Histogram
            if macd_hist > 0 and macd_hist > result['macd_hist'].iloc[-2]:
                momentum_indicators.append(1)  # Histogramme croissant (haussier)
            elif macd_hist < 0 and macd_hist < result['macd_hist'].iloc[-2]:
                momentum_indicators.append(-1)  # Histogramme décroissant (baissier)
            
            # Stochastique
            if stoch_k > 80 and stoch_d > 80:
                momentum_indicators.append(-1)  # Survente
                summary["signals"].append("Stochastic overbought")
            elif stoch_k < 20 and stoch_d < 20:
                momentum_indicators.append(1)  # Surachat
                summary["signals"].append("Stochastic oversold")
            elif stoch_k > stoch_d:
                momentum_indicators.append(0.5)  # Légèrement haussier
            else:
                momentum_indicators.append(-0.5)  # Légèrement baissier
            
            # Calculer le score de momentum
            momentum_score = sum(momentum_indicators)
            
            if momentum_score > 2:
                momentum_state = "strongly_bullish"
                momentum_strength = min(100, int(momentum_score * 25))
                momentum_description = "Strong bullish momentum"
            elif momentum_score > 0:
                momentum_state = "bullish"
                momentum_strength = min(100, int(momentum_score * 25))
                momentum_description = "Bullish momentum"
            elif momentum_score < -2:
                momentum_state = "strongly_bearish"
                momentum_strength = min(100, int(abs(momentum_score) * 25))
                momentum_description = "Strong bearish momentum"
            elif momentum_score < 0:
                momentum_state = "bearish"
                momentum_strength = min(100, int(abs(momentum_score) * 25))
                momentum_description = "Bearish momentum"
            else:
                momentum_state = "neutral"
                momentum_strength = 0
                momentum_description = "No clear momentum"
            
            summary["momentum"]["state"] = momentum_state
            summary["momentum"]["strength"] = momentum_strength
            summary["momentum"]["description"] = momentum_description
            
            # 4. Analyser la volatilité
            # ATR
            atr = result['atr'].iloc[-1]
            atr_percent = (atr / price) * 100
            
            # Bandes de Bollinger
            bb_width = result['bb_width'].iloc[-1]
            avg_bb_width = result['bb_width'].iloc[-20:].mean()
            
            # Déterminer le niveau de volatilité
            if atr_percent > 5 or bb_width > avg_bb_width * 1.5:
                volatility_level = "high"
                volatility_description = "High volatility - risk of rapid price movements"
            elif atr_percent < 1 or bb_width < avg_bb_width * 0.7:
                volatility_level = "low"
                volatility_description = "Low volatility - potential for breakout"
                
                # Vérifier s'il y a une compression des bandes de Bollinger
                if result['bb_squeeze'].iloc[-1]:
                    summary["signals"].append("Bollinger Bands squeeze - potential breakout")
            else:
                volatility_level = "moderate"
                volatility_description = "Moderate volatility"
            
            summary["volatility"]["level"] = volatility_level
            summary["volatility"]["description"] = volatility_description
            
            # 5. Analyser les niveaux de support et résistance
            sr_levels = TechnicalAnalysis.detect_support_resistance(result)
            nearest_support = sr_levels.get("nearest_support")
            nearest_resistance = sr_levels.get("nearest_resistance")
            current_price = sr_levels.get("current_price")
            
            if nearest_support:
                distance_to_support = (current_price - nearest_support) / current_price * 100
            else:
                distance_to_support = None
            
            if nearest_resistance:
                distance_to_resistance = (nearest_resistance - current_price) / current_price * 100
            else:
                distance_to_resistance = None
            
            summary["support_resistance"]["nearest_support"] = nearest_support
            summary["support_resistance"]["nearest_resistance"] = nearest_resistance
            summary["support_resistance"]["distance_to_support"] = distance_to_support
            summary["support_resistance"]["distance_to_resistance"] = distance_to_resistance
            
            # 6. Détecter les patterns de chandeliers
            candle_patterns = TechnicalAnalysis.add_candlestick_patterns(result)
            pattern_score = candle_patterns['candle_pattern_score'].iloc[-1]
            
            if pattern_score > 0:
                summary["patterns"].append("Bullish candlestick pattern detected")
            elif pattern_score < 0:
                summary["patterns"].append("Bearish candlestick pattern detected")
            
            # 7. Déterminer le biais global
            if trend_direction == "bullish" and momentum_state in ["bullish", "strongly_bullish"]:
                overall_bias = "bullish"
            elif trend_direction == "bearish" and momentum_state in ["bearish", "strongly_bearish"]:
                overall_bias = "bearish"
            else:
                overall_bias = "neutral"
            
            summary["overall_bias"] = overall_bias
        
        except Exception as e:
            summary["error"] = f"Error in technical summary: {str(e)}"
        
        return summary
    
    @staticmethod
    def analyze_price_action(df: pd.DataFrame, window: int = 10) -> Dict:
        """
        Analyse l'action des prix (price action) pour identifier des configurations intéressantes
        
        Args:
            df: DataFrame avec les données OHLCV
            window: Fenêtre d'analyse 
            
        Returns:
            Dictionnaire avec l'analyse price action
        """
        data = df.copy()
        
        # Résultat
        result = {
            "patterns": [],
            "strength": 0,
            "bias": "neutral",
            "key_levels": []
        }
        
        # Vérifier que nous avons suffisamment de données
        if len(data) < window + 5:
            result["error"] = f"Insufficient data, need at least {window + 5} bars"
            return result
        
        try:
            # 1. Analyser les corps de chandeliers
            data['body_size'] = abs(data['close'] - data['open'])
            data['body_size_pct'] = data['body_size'] / data['open'] * 100
            data['upper_wick'] = data['high'] - data[['open', 'close']].max(axis=1)
            data['lower_wick'] = data[['open', 'close']].min(axis=1) - data['low']
            data['is_bullish'] = data['close'] > data['open']
            data['is_bearish'] = data['close'] < data['open']
            
            # 2. Détecter les chandeliers importants
            # Grosses chandelles (corps > 2x la moyenne)
            avg_body_size_pct = data['body_size_pct'].rolling(window=window).mean().iloc[-1]
            latest_body_size_pct = data['body_size_pct'].iloc[-1]
            
            if latest_body_size_pct > 2 * avg_body_size_pct:
                if data['is_bullish'].iloc[-1]:
                    result["patterns"].append("Large bullish candle")
                    result["strength"] += 1
                else:
                    result["patterns"].append("Large bearish candle")
                    result["strength"] -= 1
            
            # 3. Détecter les configurations de price action
            # Outside bar (englobe la chandelle précédente)
            if (data['high'].iloc[-1] > data['high'].iloc[-2] and 
                data['low'].iloc[-1] < data['low'].iloc[-2]):
                if data['is_bullish'].iloc[-1]:
                    result["patterns"].append("Bullish outside bar")
                    result["strength"] += 1
                else:
                    result["patterns"].append("Bearish outside bar")
                    result["strength"] -= 1
            
            # Inside bar (contenue dans la chandelle précédente)
            if (data['high'].iloc[-1] < data['high'].iloc[-2] and 
                data['low'].iloc[-1] > data['low'].iloc[-2]):
                result["patterns"].append("Inside bar - potential breakout")
            
            # Pin bar (longue mèche dans une direction)
            if data['is_bullish'].iloc[-1]:
                # Bullish pin bar (longue mèche inférieure)
                if data['lower_wick'].iloc[-1] > 2 * data['body_size'].iloc[-1]:
                    result["patterns"].append("Bullish pin bar")
                    result["strength"] += 1
            else:
                # Bearish pin bar (longue mèche supérieure)
                if data['upper_wick'].iloc[-1] > 2 * data['body_size'].iloc[-1]:
                    result["patterns"].append("Bearish pin bar")
                    result["strength"] -= 1
            
            # 4. Analyse des pivots récents
            highs = data['high'].iloc[-window:].values
            lows = data['low'].iloc[-window:].values
            
            # Détecter les niveaux de swing (pivots)
            pivot_high_idx = signal.argrelextrema(highs, np.greater)[0]
            pivot_low_idx = signal.argrelextrema(lows, np.less)[0]
            
            # Collecter les niveaux de swing
            pivot_highs = [float(highs[i]) for i in pivot_high_idx]
            pivot_lows = [float(lows[i]) for i in pivot_low_idx]
            
            # 5. Détecter des structures de prix
            # Higher highs, Higher lows (tendance haussière)
            if len(pivot_highs) >= 2 and len(pivot_lows) >= 2:
                if pivot_highs[-1] > pivot_highs[-2] and pivot_lows[-1] > pivot_lows[-2]:
                    result["patterns"].append("Higher highs, higher lows - bullish structure")
                    result["bias"] = "bullish"
                    result["strength"] += 2
                # Lower highs, Lower lows (tendance baissière)
                elif pivot_highs[-1] < pivot_highs[-2] and pivot_lows[-1] < pivot_lows[-2]:
                    result["patterns"].append("Lower highs, lower lows - bearish structure")
                    result["bias"] = "bearish"
                    result["strength"] -= 2
            
            # 6. Analyser les volumes
            # Divergence volume-prix
            if (data['close'].iloc[-1] > data['close'].iloc[-2] and 
                data['volume'].iloc[-1] < data['volume'].iloc[-2]):
                result["patterns"].append("Price up, volume down - potential weakness")
            elif (data['close'].iloc[-1] < data['close'].iloc[-2] and 
                  data['volume'].iloc[-1] < data['volume'].iloc[-2]):
                result["patterns"].append("Price down, volume down - potential support")
            
            # 7. Déterminer la force et le biais global
            if result["strength"] > 2:
                result["bias"] = "strongly_bullish"
            elif result["strength"] > 0:
                result["bias"] = "bullish"
            elif result["strength"] < -2:
                result["bias"] = "strongly_bearish"
            elif result["strength"] < 0:
                result["bias"] = "bearish"
            
            # 8. Ajouter les niveaux clés
            result["key_levels"] = sorted(pivot_highs + pivot_lows)
        
        except Exception as e:
            result["error"] = f"Error in price action analysis: {str(e)}"
        
        return result

    @staticmethod
    def detect_chart_patterns(df: pd.DataFrame) -> Dict:
        """
        Détecte les patterns chartistes classiques (triangles, drapeaux, etc.)
        
        Args:
            df: DataFrame avec les données OHLCV
            
        Returns:
            Dictionnaire avec les patterns chartistes détectés
        """
        data = df.copy()
        
        result = {
            "patterns": [],
            "details": {}
        }
        
        # Vérifier que nous avons suffisamment de données
        if len(data) < 30:
            result["error"] = "Insufficient data for pattern detection"
            return result
        
        try:
            # 1. Détecter les pivots haut et bas
            window = 5  # Fenêtre pour les pivots
            data['pivot_high'] = np.nan
            data['pivot_low'] = np.nan
            
            for i in range(window, len(data) - window):
                # Un pivot haut est un point où le prix est plus haut que les 'window' points avant et après
                if all(data['high'].iloc[i] > data['high'].iloc[i-j] for j in range(1, window+1)) and \
                   all(data['high'].iloc[i] > data['high'].iloc[i+j] for j in range(1, window+1)):
                    data.loc[data.index[i], 'pivot_high'] = data['high'].iloc[i]
                
                # Un pivot bas est un point où le prix est plus bas que les 'window' points avant et après
                if all(data['low'].iloc[i] < data['low'].iloc[i-j] for j in range(1, window+1)) and \
                   all(data['low'].iloc[i] < data['low'].iloc[i+j] for j in range(1, window+1)):
                    data.loc[data.index[i], 'pivot_low'] = data['low'].iloc[i]
            
            # 2. Extraire les pivots valides (non-NaN)
            pivot_highs = data[data['pivot_high'].notnull()]
            pivot_lows = data[data['pivot_low'].notnull()]
            
            # 3. Détecter les patterns basés sur les pivots
            
            # Double Top
            if len(pivot_highs) >= 2:
                last_two_highs = pivot_highs['pivot_high'].iloc[-2:].values
                if abs(last_two_highs[0] - last_two_highs[1]) / last_two_highs[0] < 0.01:  # 1% de différence
                    result["patterns"].append("Double Top")
                    result["details"]["double_top"] = {
                        "level": float(np.mean(last_two_highs)),
                        "strength": "high" if data['volume'].iloc[-1] > data['volume'].rolling(20).mean().iloc[-1] else "medium"
                    }
            
            # Double Bottom
            if len(pivot_lows) >= 2:
                last_two_lows = pivot_lows['pivot_low'].iloc[-2:].values
                if abs(last_two_lows[0] - last_two_lows[1]) / last_two_lows[0] < 0.01:  # 1% de différence
                    result["patterns"].append("Double Bottom")
                    result["details"]["double_bottom"] = {
                        "level": float(np.mean(last_two_lows)),
                        "strength": "high" if data['volume'].iloc[-1] > data['volume'].rolling(20).mean().iloc[-1] else "medium"
                    }
            
            # Head and Shoulders (besoin de 3 pivots hauts avec celui du milieu plus haut)
            if len(pivot_highs) >= 3:
                last_three_highs = pivot_highs['pivot_high'].iloc[-3:].values
                if last_three_highs[1] > last_three_highs[0] and last_three_highs[1] > last_three_highs[2] and \
                   abs(last_three_highs[0] - last_three_highs[2]) / last_three_highs[0] < 0.05:  # Épaules similaires
                    result["patterns"].append("Head and Shoulders")
                    result["details"]["head_and_shoulders"] = {
                        "neckline": float(pivot_lows['pivot_low'].iloc[-2:].min()),
                        "strength": "high"
                    }
            
            # 4. Détecter les triangles et les canaux
            
            # Triangle ascendant (bas plats, hauts descendants)
            if len(pivot_lows) >= 3 and len(pivot_highs) >= 3:
                # Vérifier si les bas sont relativement plats
                low_std = np.std(pivot_lows['pivot_low'].iloc[-3:].values)
                low_mean = np.mean(pivot_lows['pivot_low'].iloc[-3:].values)
                
                # Vérifier si les hauts sont descendants
                highs = pivot_highs['pivot_high'].iloc[-3:].values
                descending_highs = all(highs[i] > highs[i+1] for i in range(len(highs)-1))
                
                if low_std / low_mean < 0.01 and descending_highs:  # Bas plats (variance < 1%) et hauts descendants
                    result["patterns"].append("Ascending Triangle")
                    result["details"]["ascending_triangle"] = {
                        "resistance": float(pivot_highs['pivot_high'].iloc[-1]),
                        "support": float(low_mean)
                    }
            
            # Triangle descendant (hauts plats, bas ascendants)
            if len(pivot_lows) >= 3 and len(pivot_highs) >= 3:
                # Vérifier si les hauts sont relativement plats
                high_std = np.std(pivot_highs['pivot_high'].iloc[-3:].values)
                high_mean = np.mean(pivot_highs['pivot_high'].iloc[-3:].values)
                
                # Vérifier si les bas sont ascendants
                lows = pivot_lows['pivot_low'].iloc[-3:].values
                ascending_lows = all(lows[i] < lows[i+1] for i in range(len(lows)-1))
                
                if high_std / high_mean < 0.01 and ascending_lows:  # Hauts plats et bas ascendants
                    result["patterns"].append("Descending Triangle")
                    result["details"]["descending_triangle"] = {
                        "resistance": float(high_mean),
                        "support": float(pivot_lows['pivot_low'].iloc[-1])
                    }
            
              # 5. Détecter les canaux
            if len(pivot_highs) >= 3 and len(pivot_lows) >= 3:
                # Canal ascendant
                high_vals = pivot_highs['pivot_high'].iloc[-3:].values
                low_vals = pivot_lows['pivot_low'].iloc[-3:].values
                
                ascending_highs = all(high_vals[i] < high_vals[i+1] for i in range(len(high_vals)-1))
                ascending_lows = all(low_vals[i] < low_vals[i+1] for i in range(len(low_vals)-1))
                
                if ascending_highs and ascending_lows:
                    result["patterns"].append("Ascending Channel")
                    result["details"]["ascending_channel"] = {
                        "slope": float((high_vals[-1] - high_vals[0]) / len(high_vals)),
                        "upper": float(high_vals[-1]),
                        "lower": float(low_vals[-1])
                    }
                
                # Canal descendant
                descending_highs = all(high_vals[i] > high_vals[i+1] for i in range(len(high_vals)-1))
                descending_lows = all(low_vals[i] > low_vals[i+1] for i in range(len(low_vals)-1))
                
                if descending_highs and descending_lows:
                    result["patterns"].append("Descending Channel")
                    result["details"]["descending_channel"] = {
                        "slope": float((high_vals[0] - high_vals[-1]) / len(high_vals)),
                        "upper": float(high_vals[-1]),
                        "lower": float(low_vals[-1])
                    }
      
        except Exception as e:
            result["error"] = f"Error in chart pattern detection: {str(e)}"
        
        return result

    @staticmethod
    def calculate_fibonacci_levels(df: pd.DataFrame, direction: str = 'auto') -> Dict:
        """
        Calcule les niveaux de Fibonacci basés sur le mouvement récent des prix
        
        Args:
            df: DataFrame avec les données OHLCV
            direction: Direction du mouvement ('up', 'down', ou 'auto' pour détection automatique)
            
        Returns:
            Dictionnaire avec les niveaux de Fibonacci
        """
        data = df.copy()
        result = {
            "levels": {},
            "direction": direction
        }
        
        # Vérifier que nous avons suffisamment de données
        if len(data) < 20:
            result["error"] = "Insufficient data for Fibonacci calculations"
            return result
        
        try:
            # 1. Déterminer la direction si 'auto'
            if direction == 'auto':
                # Calculer la tendance sur les 20 dernières barres
                trend = data['close'].iloc[-1] - data['close'].iloc[-20]
                direction = 'up' if trend > 0 else 'down'
                result["direction"] = direction
            
            # 2. Identifier les points de swing pour le calcul des niveaux
            if direction == 'up':
                # Pour une tendance haussière, prendre le plus bas récent et le plus haut récent
                recent_low = data['low'].iloc[-20:].min()
                recent_high = data['high'].iloc[-5:].max()
                
                # Vérifier que le low est avant le high
                low_idx = data['low'].iloc[-20:].idxmin()
                high_idx = data['high'].iloc[-5:].idxmax()
                
                if low_idx >= high_idx:
                    # Chercher un minimum plus ancien
                    recent_low = data['low'].iloc[-50:-20].min()
                    low_idx = data['low'].iloc[-50:-20].idxmin()
            else:
                # Pour une tendance baissière, prendre le plus haut récent et le plus bas récent
                recent_high = data['high'].iloc[-20:].max()
                recent_low = data['low'].iloc[-5:].min()
                
                # Vérifier que le high est avant le low
                high_idx = data['high'].iloc[-20:].idxmax()
                low_idx = data['low'].iloc[-5:].idxmin()
                
                if high_idx >= low_idx:
                    # Chercher un maximum plus ancien
                    recent_high = data['high'].iloc[-50:-20].max()
                    high_idx = data['high'].iloc[-50:-20].idxmax()
            
            # 3. Calculer les niveaux de Fibonacci
            diff = abs(recent_high - recent_low)
            
            if direction == 'up':
                # Niveaux de retracement pour tendance haussière
                result["levels"] = {
                    "0.0": float(recent_high),
                    "0.236": float(recent_high - 0.236 * diff),
                    "0.382": float(recent_high - 0.382 * diff),
                    "0.5": float(recent_high - 0.5 * diff),
                    "0.618": float(recent_high - 0.618 * diff),
                    "0.786": float(recent_high - 0.786 * diff),
                    "1.0": float(recent_low),
                    # Extensions
                    "1.618": float(recent_high + 0.618 * diff),
                    "2.618": float(recent_high + 1.618 * diff)
                }
            else:
                # Niveaux de retracement pour tendance baissière
                result["levels"] = {
                    "0.0": float(recent_low),
                    "0.236": float(recent_low + 0.236 * diff),
                    "0.382": float(recent_low + 0.382 * diff),
                    "0.5": float(recent_low + 0.5 * diff),
                    "0.618": float(recent_low + 0.618 * diff),
                    "0.786": float(recent_low + 0.786 * diff),
                    "1.0": float(recent_high),
                    # Extensions
                    "1.618": float(recent_low - 0.618 * diff),
                    "2.618": float(recent_low - 1.618 * diff)
                }
            
            # Ajouter les informations sur le swing
            result["swing"] = {
                "low": float(recent_low),
                "high": float(recent_high),
                "diff": float(diff),
                "low_date": str(low_idx),
                "high_date": str(high_idx)
            }
            
            # 4. Identifier les niveaux à surveiller (les plus proches du prix actuel)
            current_price = data['close'].iloc[-1]
            levels_list = [(k, v) for k, v in result["levels"].items()]
            levels_list.sort(key=lambda x: abs(x[1] - current_price))
            
            result["nearest_levels"] = [
                {"ratio": levels_list[0][0], "price": levels_list[0][1], "distance_pct": abs(levels_list[0][1] - current_price) / current_price * 100},
                {"ratio": levels_list[1][0], "price": levels_list[1][1], "distance_pct": abs(levels_list[1][1] - current_price) / current_price * 100}
            ]
            
        except Exception as e:
            result["error"] = f"Error in Fibonacci calculation: {str(e)}"
        
        return result