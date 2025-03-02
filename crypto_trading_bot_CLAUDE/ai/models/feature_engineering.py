# ai/models/feature_engineering.py
"""
Module d'ingénierie des caractéristiques pour le modèle LSTM
Prépare les données brutes pour l'entraînement et la prédiction
"""
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Union, Optional
from datetime import datetime, timedelta
import talib
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import pickle
import os

from indicators.trend import calculate_ema, calculate_adx, calculate_macd
from indicators.momentum import calculate_rsi, calculate_stochastic
from indicators.volatility import calculate_bollinger_bands, calculate_atr
from indicators.volume import calculate_obv, calculate_vwap
from config.config import DATA_DIR
from utils.logger import setup_logger

logger = setup_logger("feature_engineering")

class FeatureEngineering:
    """
    Classe pour la création et transformation des caractéristiques
    pour l'entraînement du modèle LSTM
    """
    def __init__(self, save_scalers: bool = True):
        """
        Initialise le module d'ingénierie des caractéristiques
        
        Args:
            save_scalers: Indique s'il faut sauvegarder les scaler pour réutilisation
        """
        self.save_scalers = save_scalers
        self.scalers = {}
        self.scalers_path = os.path.join(DATA_DIR, "models", "scalers")
        
        # Créer le répertoire pour les scalers si nécessaire
        if save_scalers and not os.path.exists(self.scalers_path):
            os.makedirs(self.scalers_path, exist_ok=True)
    
    def create_features(self, data: pd.DataFrame, 
                     include_time_features: bool = True,
                     include_price_patterns: bool = True) -> pd.DataFrame:
        """
        Crée des caractéristiques avancées à partir des données OHLCV
        
        Args:
            data: DataFrame avec au moins les colonnes OHLCV (open, high, low, close, volume)
            include_time_features: Inclure les caractéristiques temporelles
            include_price_patterns: Inclure la détection des patterns de prix
            
        Returns:
            DataFrame enrichi avec les caractéristiques créées
        """
        # Vérifier que les colonnes requises sont présentes
        required_columns = ['open', 'high', 'low', 'close', 'volume']
        if not all(col in data.columns for col in required_columns):
            raise ValueError(f"Colonnes requises manquantes. Nécessite: {required_columns}")
        
        # Copier le DataFrame pour éviter de modifier l'original
        df = data.copy()
        
        # Assurer que l'index est un DatetimeIndex pour les caractéristiques temporelles
        if include_time_features and not isinstance(df.index, pd.DatetimeIndex):
            if 'timestamp' in df.columns:
                df['timestamp'] = pd.to_datetime(df['timestamp'])
                df.set_index('timestamp', inplace=True)
            else:
                logger.warning("Impossible de créer des caractéristiques temporelles sans colonne timestamp")
                include_time_features = False
        
        # 1. Indicateurs de tendance
        # EMA à différentes périodes
        ema_periods = [9, 21, 50, 200]
        emas = calculate_ema(df, ema_periods)
        for period, ema_series in emas.items():
            df[f'{period}'] = ema_series
        
        # Distances relatives aux EMAs
        for period in ema_periods:
            df[f'dist_to_ema_{period}'] = (df['close'] - df[f'ema_{period}']) / df[f'ema_{period}'] * 100
        
        # MACD
        macd_data = calculate_macd(df)
        df['macd'] = macd_data['macd']
        df['macd_signal'] = macd_data['signal']
        df['macd_hist'] = macd_data['histogram']
        
        # ADX (force de tendance)
        adx_data = calculate_adx(df)
        df['adx'] = adx_data['adx']
        df['plus_di'] = adx_data['plus_di']
        df['minus_di'] = adx_data['minus_di']
        
        # 2. Indicateurs de momentum
        # RSI
        df['rsi'] = calculate_rsi(df)
        
        # Stochastique
        stoch_data = calculate_stochastic(df)
        df['stoch_k'] = stoch_data['k']
        df['stoch_d'] = stoch_data['d']
        
        # Rate of Change (ROC) à différentes périodes
        for period in [5, 10, 21]:
            df[f'roc_{period}'] = df['close'].pct_change(period) * 100
        
        # 3. Indicateurs de volatilité
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
        
        # 4. Indicateurs de volume
        # OBV (On-Balance Volume)
        df['obv'] = calculate_obv(df)
        
        # Volume relatif (comparé à la moyenne)
        for period in [5, 10, 21]:
            df[f'rel_volume_{period}'] = df['volume'] / df['volume'].rolling(period).mean()
        
        # VWAP (Volume-Weighted Average Price)
        df['vwap'] = calculate_vwap(df)
        df['vwap_dist'] = (df['close'] - df['vwap']) / df['vwap'] * 100  # Distance au VWAP
        
        # 5. Caractéristiques de prix
        # Rendements à différentes périodes
        for period in [1, 3, 5, 10]:
            df[f'return_{period}'] = df['close'].pct_change(period) * 100
        
        # Caractéristiques des chandeliers
        df['body_size'] = abs(df['close'] - df['open'])
        df['body_size_percent'] = df['body_size'] / df['open'] * 100
        df['upper_wick'] = df['high'] - df[['open', 'close']].max(axis=1)
        df['lower_wick'] = df[['open', 'close']].min(axis=1) - df['low']
        df['upper_wick_percent'] = df['upper_wick'] / df['open'] * 100
        df['lower_wick_percent'] = df['lower_wick'] / df['open'] * 100
        
        # Détection des gaps
        df['gap_up'] = (df['low'] > df['high'].shift(1)).astype(int)
        df['gap_down'] = (df['high'] < df['low'].shift(1)).astype(int)
        
        # 6. Caractéristiques temporelles
        if include_time_features:
            # Heure de la journée (valeurs cycliques sin/cos)
            hour = df.index.hour
            df['hour_sin'] = np.sin(2 * np.pi * hour / 24)
            df['hour_cos'] = np.cos(2 * np.pi * hour / 24)
            
            # Jour de la semaine (valeurs cycliques sin/cos)
            day_of_week = df.index.dayofweek
            df['day_sin'] = np.sin(2 * np.pi * day_of_week / 7)
            df['day_cos'] = np.cos(2 * np.pi * day_of_week / 7)
            
            # Jour du mois (valeurs cycliques sin/cos)
            day = df.index.day
            df['day_of_month_sin'] = np.sin(2 * np.pi * day / 31)
            df['day_of_month_cos'] = np.cos(2 * np.pi * day / 31)
        
        # 7. Détection des patterns de prix (via talib)
        if include_price_patterns:
            try:
                # Patterns de retournement haussier
                df['hammer'] = talib.CDLHAMMER(df['open'].values, df['high'].values, 
                                               df['low'].values, df['close'].values)
                df['inverted_hammer'] = talib.CDLINVERTEDHAMMER(df['open'].values, df['high'].values, 
                                                               df['low'].values, df['close'].values)
                df['morning_star'] = talib.CDLMORNINGSTAR(df['open'].values, df['high'].values, 
                                                         df['low'].values, df['close'].values)
                df['bullish_engulfing'] = talib.CDLENGULFING(df['open'].values, df['high'].values, 
                                                            df['low'].values, df['close'].values)
                
                # Patterns de retournement baissier
                df['shooting_star'] = talib.CDLSHOOTINGSTAR(df['open'].values, df['high'].values, 
                                                           df['low'].values, df['close'].values)
                df['evening_star'] = talib.CDLEVENINGSTAR(df['open'].values, df['high'].values, 
                                                         df['low'].values, df['close'].values)
                df['bearish_engulfing'] = -talib.CDLENGULFING(df['open'].values, df['high'].values, 
                                                             df['low'].values, df['close'].values)
                
                # Créer une caractéristique résumée des patterns
                bullish_patterns = df[['hammer', 'inverted_hammer', 'morning_star', 'bullish_engulfing']].sum(axis=1)
                bearish_patterns = df[['shooting_star', 'evening_star', 'bearish_engulfing']].sum(axis=1)
                
                df['bullish_patterns'] = bullish_patterns
                df['bearish_patterns'] = bearish_patterns
                
            except Exception as e:
                logger.warning(f"Erreur lors de la détection des patterns de prix: {str(e)}")
        
        # 8. Caractéristiques de support/résistance
        # Identifier les niveaux de support/résistance majeurs sur 50 périodes
        window = 50
        if len(df) >= window:
            # Détecter les sommets locaux (hauts)
            df['is_high'] = (df['high'] > df['high'].shift(1)) & (df['high'] > df['high'].shift(-1))
            
            # Détecter les creux locaux (bas)
            df['is_low'] = (df['low'] < df['low'].shift(1)) & (df['low'] < df['low'].shift(-1))
            
            # Distance par rapport au plus haut récent
            rolling_high = df['high'].rolling(window).max()
            df['dist_to_high'] = (df['close'] - rolling_high) / rolling_high * 100
            
            # Distance par rapport au plus bas récent
            rolling_low = df['low'].rolling(window).min()
            df['dist_to_low'] = (df['close'] - rolling_low) / rolling_low * 100
        
        # 9. Caractéristiques croisées
        # RSI contre les bandes de Bollinger
        df['rsi_bb'] = (df['rsi'] - 50) * df['bb_percent_b']
        
        # Momentum de prix et volume
        df['price_volume_trend'] = df['return_1'] * df['rel_volume_5']
        
        # Caractéristique d'inversion de tendance (combo ADX + RSI)
        df['reversal_signal'] = ((df['adx'] > 25) & (df['rsi'] < 30)) | ((df['adx'] > 25) & (df['rsi'] > 70))
        
        # 10. Nettoyer les données
        # Remplacer les valeurs infinies
        df.replace([np.inf, -np.inf], np.nan, inplace=True)
        
        # Supprimer les colonnes avec trop de NaN
        threshold = len(df) * 0.9  # 90% des valeurs doivent être non-NA
        df = df.dropna(axis=1, thresh=threshold)
        
        # Remplir les NaN restants avec des valeurs appropriées
        df.fillna(method='ffill', inplace=True)  # Forward fill
        df.fillna(0, inplace=True)  # Remplacer les NaN restants par 0
        
        return df
    
    def scale_features(self, data: pd.DataFrame, is_training: bool = True,
                     method: str = 'standard', feature_group: str = 'default') -> pd.DataFrame:
        """
        Normalise les caractéristiques pour l'entraînement du modèle
        
        Args:
            data: DataFrame avec les caractéristiques
            is_training: Indique si c'est pour l'entraînement ou la prédiction
            method: Méthode de scaling ('standard' ou 'minmax')
            feature_group: Groupe de caractéristiques pour sauvegarder/charger les scalers
            
        Returns:
            DataFrame avec les caractéristiques normalisées
        """
        # Sélectionner les colonnes numériques
        numeric_cols = data.select_dtypes(include=['float64', 'int64']).columns.tolist()
        
        # Exclure les colonnes qu'on ne veut pas normaliser
        cols_to_exclude = ['timestamp', 'date', 'open', 'high', 'low', 'close', 'volume']
        feature_cols = [col for col in numeric_cols if col not in cols_to_exclude]
        
        # Si aucune caractéristique à normaliser, retourner les données telles quelles
        if not feature_cols:
            logger.warning("Aucune caractéristique à normaliser")
            return data
        
        # Pour l'entraînement, créer et ajuster de nouveaux scalers
        if is_training:
            if method == 'standard':
                scaler = StandardScaler()
            else:  # 'minmax'
                scaler = MinMaxScaler(feature_range=(-1, 1))
            
            # Ajuster le scaler sur les données d'entraînement
            scaler.fit(data[feature_cols])
            
            # Sauvegarder le scaler
            if self.save_scalers:
                scaler_path = os.path.join(self.scalers_path, f"{feature_group}_{method}_scaler.pkl")
                with open(scaler_path, 'wb') as f:
                    pickle.dump(scaler, f)
            
            # Stocker le scaler en mémoire
            self.scalers[f"{feature_group}_{method}"] = scaler
        
        # Pour la prédiction, utiliser un scaler existant
        else:
            scaler_key = f"{feature_group}_{method}"
            
            # Chercher d'abord en mémoire
            if scaler_key in self.scalers:
                scaler = self.scalers[scaler_key]
            
            # Sinon, charger depuis le disque
            else:
                scaler_path = os.path.join(self.scalers_path, f"{feature_group}_{method}_scaler.pkl")
                if os.path.exists(scaler_path):
                    with open(scaler_path, 'rb') as f:
                        scaler = pickle.load(f)
                    # Stocker en mémoire pour usage futur
                    self.scalers[scaler_key] = scaler
                else:
                    logger.error(f"Scaler non trouvé pour la prédiction: {scaler_key}")
                    raise FileNotFoundError(f"Scaler non trouvé: {scaler_path}")
        
        # Transformer les données
        scaled_features = scaler.transform(data[feature_cols])
        
        # Créer un nouveau DataFrame avec les caractéristiques normalisées
        scaled_df = data.copy()
        scaled_df[feature_cols] = scaled_features
        
        return scaled_df
    
    def prepare_lstm_data(self, data: pd.DataFrame, sequence_length: int = 60,
                         prediction_horizon: int = 12, is_training: bool = True) -> Tuple:
        """
        Prépare les données au format requis par le modèle LSTM
        
        Args:
            data: DataFrame avec les caractéristiques (déjà normalisées)
            sequence_length: Longueur des séquences d'entrée
            prediction_horizon: Horizon de prédiction (nombre de périodes)
            is_training: Indique si c'est pour l'entraînement ou la prédiction
            
        Returns:
            Tuple (X, y) pour l'entraînement ou X pour la prédiction
        """
        # Sélectionner les colonnes de caractéristiques (exclure timestamp/date/etc.)
        feature_cols = data.select_dtypes(include=['float64', 'int64']).columns.tolist()
        
        # Exclure des colonnes spécifiques si nécessaires
        cols_to_exclude = ['timestamp', 'date']
        feature_cols = [col for col in feature_cols if col not in cols_to_exclude]
        
        # Créer les séquences d'entrée
        X = []
        
        for i in range(len(data) - sequence_length - (prediction_horizon if is_training else 0)):
            X.append(data[feature_cols].iloc[i:i+sequence_length].values)
        
        X = np.array(X)
        
        # Si c'est pour la prédiction, retourner seulement X
        if not is_training:
            return X
        
        # Sinon, créer également les labels
        y_direction = []
        y_volatility = []
        y_volume = []
        y_momentum = []
        
        for i in range(len(data) - sequence_length - prediction_horizon):
            # Prix actuel (à la fin de la séquence d'entrée)
            current_price = data['close'].iloc[i+sequence_length-1]
            
            # Prix futur (après l'horizon de prédiction)
            future_price = data['close'].iloc[i+sequence_length+prediction_horizon-1]
            
            # Direction (1 si hausse, 0 si baisse)
            direction = 1 if future_price > current_price else 0
            y_direction.append(direction)
            
            # Volatilité (écart-type des rendements futurs)
            future_returns = data['close'].iloc[i+sequence_length:i+sequence_length+prediction_horizon].pct_change().dropna()
            volatility = future_returns.std() * np.sqrt(prediction_horizon)  # Annualisé
            y_volatility.append(volatility)
            
            # Volume relatif futur
            current_volume = data['volume'].iloc[i+sequence_length-1]
            future_volume = data['volume'].iloc[i+sequence_length:i+sequence_length+prediction_horizon].mean()
            relative_volume = future_volume / current_volume if current_volume > 0 else 1.0
            y_volume.append(relative_volume)
            
            # Momentum (changement de prix normalisé)
            price_change_pct = (future_price - current_price) / current_price
            momentum = np.tanh(price_change_pct * 5)  # Utiliser tanh pour normaliser entre -1 et 1
            y_momentum.append(momentum)
        
        # Convertir en tableaux numpy
        y_direction = np.array(y_direction)
        y_volatility = np.array(y_volatility)
        y_volume = np.array(y_volume)
        y_momentum = np.array(y_momentum)
        
        # Empaqueter dans un tuple
        y = (y_direction, y_volatility, y_volume, y_momentum)
        
        return X, y
    
    def create_multi_horizon_data(self, data: pd.DataFrame, 
                               sequence_length: int = 60,
                               horizons: List[int] = [12, 24, 96],
                               is_training: bool = True) -> Tuple:
        """
        Prépare les données pour une prédiction multi-horizon
        
        Args:
            data: DataFrame avec les caractéristiques (déjà normalisées)
            sequence_length: Longueur des séquences d'entrée
            horizons: Liste des horizons de prédiction (en périodes)
            is_training: Indique si c'est pour l'entraînement ou la prédiction
            
        Returns:
            Tuple (X, y_list) pour l'entraînement ou X pour la prédiction
        """
        # Sélectionner les colonnes de caractéristiques
        feature_cols = data.select_dtypes(include=['float64', 'int64']).columns.tolist()
        cols_to_exclude = ['timestamp', 'date']
        feature_cols = [col for col in feature_cols if col not in cols_to_exclude]
        
        # Créer les séquences d'entrée
        X = []
        
        for i in range(len(data) - sequence_length - (max(horizons) if is_training else 0)):
            X.append(data[feature_cols].iloc[i:i+sequence_length].values)
        
        X = np.array(X)
        
        # Si c'est pour la prédiction, retourner seulement X
        if not is_training:
            return X
        
        # Sinon, créer également les labels pour chaque horizon
        y_list = []
        
        for horizon in horizons:
            y_direction = []
            y_volatility = []
            y_volume = []
            y_momentum = []
            
            for i in range(len(data) - sequence_length - horizon):
                # Prix actuel (à la fin de la séquence d'entrée)
                current_price = data['close'].iloc[i+sequence_length-1]
                
                # Prix futur (après l'horizon de prédiction)
                future_price = data['close'].iloc[i+sequence_length+horizon-1]
                
                # Direction (1 si hausse, 0 si baisse)
                direction = 1 if future_price > current_price else 0
                y_direction.append(direction)
                
                # Volatilité (écart-type des rendements futurs)
                future_returns = data['close'].iloc[i+sequence_length:i+sequence_length+horizon].pct_change().dropna()
                volatility = future_returns.std() * np.sqrt(horizon)  # Annualisé
                y_volatility.append(volatility)
                
                # Volume relatif futur
                current_volume = data['volume'].iloc[i+sequence_length-1]
                future_volume = data['volume'].iloc[i+sequence_length:i+sequence_length+horizon].mean()
                relative_volume = future_volume / current_volume if current_volume > 0 else 1.0
                y_volume.append(relative_volume)
                
                # Momentum (changement de prix normalisé)
                price_change_pct = (future_price - current_price) / current_price
                momentum = np.tanh(price_change_pct * 5)
                y_momentum.append(momentum)
            
            # Convertir en tableaux numpy
            y_direction = np.array(y_direction)
            y_volatility = np.array(y_volatility)
            y_volume = np.array(y_volume)
            y_momentum = np.array(y_momentum)
            
            # Ajouter à la liste des sorties
            y_list.extend([y_direction, y_volatility, y_volume, y_momentum])
        
        return X, y_list