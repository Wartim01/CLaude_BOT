# ai/models/feature_engineering.py
"""
Module d'ingénierie des caractéristiques pour le modèle LSTM
Prépare les données brutes pour l'entraînement et la prédiction
"""
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Union, Optional
from datetime import datetime, timedelta
import sys  # Ajout de l'import manquant
try:
    import talib
except ImportError as e:
    raise ImportError("Module 'talib' is required. Please install it using 'pip install TA-Lib'") from e
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score
import pickle
import os
import json

from indicators.trend import calculate_ema, calculate_adx, calculate_macd
from indicators.momentum import calculate_rsi, calculate_stochastic
from indicators.volatility import calculate_bollinger_bands, calculate_atr
from indicators.volume import calculate_obv, calculate_vwap
from config.config import DATA_DIR
from utils.logger import setup_logger
from config.feature_config import get_optimal_feature_count, update_optimal_feature_count
from config.feature_config import FEATURE_COLUMNS

logger = setup_logger("feature_engineering")

class FeatureEngineering:
    """
    Classe pour la création et transformation des caractéristiques
    pour l'entraînement du modèle LSTM
    """
    def __init__(self, save_scalers: bool = True, expected_feature_count: int = None, 
                auto_optimize: bool = False):
        """
        Initialise le module d'ingénierie des caractéristiques.
        
        Args:
            save_scalers: Indique s'il faut sauvegarder les scalers pour réutilisation.
            expected_feature_count: Nombre de features attendu après traitement.
            auto_optimize: Activer l'optimisation automatique du nombre de caractéristiques.
        """
        self.save_scalers = save_scalers
        # Utiliser la valeur centralisée si aucune valeur n'est fournie
        self.expected_feature_count = expected_feature_count if expected_feature_count is not None else get_optimal_feature_count()
        self.auto_optimize = auto_optimize
        self.optimal_feature_count = None
        self.scalers = {}
        self.scalers_path = os.path.join(DATA_DIR, "models", "scalers")
        self.fixed_features = None  # Liste des colonnes à conserver de façon cohérente
        self.feature_importances = {}  # Stockage des importances de features
        
        # Créer le répertoire pour les scalers si nécessaire
        if save_scalers and not os.path.exists(self.scalers_path):
            os.makedirs(self.scalers_path, exist_ok=True)
        
        # Si auto_optimize est activé, essayez de charger la configuration optimale existante
        if auto_optimize:
            self.load_feature_configuration()

    def create_features(self, data: pd.DataFrame, 
                     include_time_features: bool = True,
                     include_price_patterns: bool = True,
                     enforce_consistency: bool = True,
                     force_feature_count: int = None) -> pd.DataFrame:
        """
        Crée des caractéristiques avancées à partir des données OHLCV
        
        Args:
            data: DataFrame avec au moins les colonnes OHLCV (open, high, low, close, volume)
            include_time_features: Inclure les caractéristiques temporelles
            include_price_patterns: Inclure la détection des patterns de prix
            enforce_consistency: Assurer la cohérence des caractéristiques entre les appels
            force_feature_count: Forcer un nombre spécifique de caractéristiques (ignorant expected_feature_count)
            
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
            df[f'ema_{period}'] = ema_series
            df[f'dist_to_ema_{period}'] = (df['close'] - ema_series) / ema_series * 100
        
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
        df.replace([np.inf, -np.inf], np.nan, inplace=True)
        # Comment out dropna to preserve all generated features
        # threshold = len(df) * 0.9  # 90% non-NA required
        # df = df.dropna(axis=1, thresh=threshold)
        df.ffill(inplace=True)
        df.fillna(0, inplace=True)
        
        logger.info(f"Nombre de features généré avant harmonisation: {df.shape[1]}")

        # Vérifier d'abord si une configuration précédente existe et doit être utilisée
        if enforce_consistency and self.fixed_features is not None:
            logger.info(f"Utilisation de la liste fixe de caractéristiques (configuration existante)")
            # Calculer les colonnes existantes vs. attendues
            existing_cols = set(df.columns)
            required_cols = set(self.fixed_features)
            
            # Colonnes manquantes et en trop
            missing_cols = required_cols - existing_cols
            extra_cols = existing_cols - required_cols
            
            if missing_cols:
                logger.warning(f"Caractéristiques manquantes: {len(missing_cols)}. Ajout de colonnes de remplacement.")
                
                # Mapping for columns that need specific calculations
                calc_map = {
                    "open": lambda d: d["open"],
                    "high": lambda d: d["high"],
                    "low": lambda d: d["low"],
                    "close": lambda d: d["close"],
                    "volume": lambda d: d["volume"],
                    "ema_9": lambda d: calculate_ema(d, [9])[9],
                    "ema_21": lambda d: calculate_ema(d, [21])[21],
                    "ema_50": lambda d: calculate_ema(d, [50])[50],
                    "ema_200": lambda d: calculate_ema(d, [200])[200],
                    "dist_to_ema_9": lambda d: (d["close"] - calculate_ema(d, [9])[9]) / calculate_ema(d, [9])[9] * 100,
                    "dist_to_ema_21": lambda d: (d["close"] - calculate_ema(d, [21])[21]) / calculate_ema(d, [21])[21] * 100,
                    "dist_to_ema_50": lambda d: (d["close"] - calculate_ema(d, [50])[50]) / calculate_ema(d, [50])[50] * 100,
                    "dist_to_ema_200": lambda d: (d["close"] - calculate_ema(d, [200])[200]) / calculate_ema(d, [200])[200] * 100,
                    "macd": lambda d: calculate_macd(d)["macd"],
                    "macd_signal": lambda d: calculate_macd(d)["signal"],
                    "macd_hist": lambda d: calculate_macd(d)["histogram"],
                    "adx": lambda d: calculate_adx(d)["adx"],
                    "plus_di": lambda d: calculate_adx(d)["plus_di"],
                    "minus_di": lambda d: calculate_adx(d)["minus_di"],
                    "rsi": lambda d: calculate_rsi(d),
                    "stoch_k": lambda d: calculate_stochastic(d)["k"],
                    "stoch_d": lambda d: calculate_stochastic(d)["d"],
                    "roc_5": lambda d: d["close"].pct_change(5) * 100,
                    "roc_10": lambda d: d["close"].pct_change(10) * 100,
                    "roc_21": lambda d: d["close"].pct_change(21) * 100,
                    "bb_upper": lambda d: calculate_bollinger_bands(d)["upper"],
                    "bb_middle": lambda d: calculate_bollinger_bands(d)["middle"],
                    "bb_lower": lambda d: calculate_bollinger_bands(d)["lower"],
                    "bb_width": lambda d: calculate_bollinger_bands(d)["bandwidth"],
                    "bb_percent_b": lambda d: calculate_bollinger_bands(d)["percent_b"],
                    "atr": lambda d: calculate_atr(d),
                    "atr_percent": lambda d: calculate_atr(d) / d["close"] * 100,
                    "obv": lambda d: calculate_obv(d),
                    "rel_volume_5": lambda d: d["volume"] / d["volume"].rolling(5).mean(),
                    "rel_volume_10": lambda d: d["volume"] / d["volume"].rolling(10).mean(),
                    "rel_volume_21": lambda d: d["volume"] / d["volume"].rolling(21).mean(),
                    "vwap": lambda d: calculate_vwap(d),
                    "vwap_dist": lambda d: (d["close"] - calculate_vwap(d)) / calculate_vwap(d) * 100,
                    "return_1": lambda d: d["close"].pct_change(1) * 100,
                    "return_3": lambda d: d["close"].pct_change(3) * 100,
                    "return_5": lambda d: d["close"].pct_change(5) * 100,
                    "return_10": lambda d: d["close"].pct_change(10) * 100,
                    "body_size": lambda d: abs(d["close"] - d["open"]),
                    "body_size_percent": lambda d: abs(d["close"] - d["open"]) / d["open"] * 100,
                    "upper_wick": lambda d: d["high"] - d[["open", "close"]].max(axis=1),
                    "lower_wick": lambda d: d[["open", "close"]].min(axis=1) - d["low"],
                    "upper_wick_percent": lambda d: (d["high"] - d[["open", "close"]].max(axis=1)) / d["open"] * 100,
                    "lower_wick_percent": lambda d: (d[["open", "close"]].min(axis=1) - d["low"]) / d["open"] * 100,
                    "gap_up": lambda d: (d["low"] > d["high"].shift(1)).astype(int),
                    "gap_down": lambda d: (d["high"] < d["low"].shift(1)).astype(int),
                    "hour_sin": lambda d: np.sin(2 * np.pi * d.index.hour / 24),
                    "hour_cos": lambda d: np.cos(2 * np.pi * d.index.hour / 24),
                    "day_sin": lambda d: np.sin(2 * np.pi * d.index.dayofweek / 7),
                    "day_cos": lambda d: np.cos(2 * np.pi * d.index.dayofweek / 7),
                    "day_of_month_sin": lambda d: np.sin(2 * np.pi * d.index.day / 31),
                    "day_of_month_cos": lambda d: np.cos(2 * np.pi * d.index.day / 31),
                    "is_high": lambda d: (d["high"] > d["high"].shift(1)) & (d["high"] > d["high"].shift(-1)),
                    "is_low": lambda d: (d["low"] < d["low"].shift(1)) & (d["low"] < d["low"].shift(-1)),
                    "dist_to_high": lambda d: (d["close"] - d["high"].rolling(50).max()) / d["high"].rolling(50).max() * 100,
                    "dist_to_low": lambda d: (d["close"] - d["low"].rolling(50).min()) / d["low"].rolling(50).min() * 100,
                    "rsi_bb": lambda d: (d["rsi"] - 50) * d["bb_percent_b"],
                    "price_volume_trend": lambda d: d["return_1"] * d["rel_volume_5"],
                    "reversal_signal": lambda d: ((d["adx"] > 25) & (d["rsi"] < 30)) | ((d["adx"] > 25) & (d["rsi"] > 70))
                }

                for col in missing_cols:
                    try:
                        if col in calc_map:
                            df[col] = calc_map[col](df)
                        else:
                            # Default filler for columns without specific logic
                            df[col] = 0
                    except Exception:
                        df[col] = 0
            
            if extra_cols:
                logger.warning(f"Caractéristiques supplémentaires ignorées: {len(extra_cols)}")
                
            # Réordonner les colonnes exactement comme dans la configuration précédente
            df = df[self.fixed_features]
            
            # Vérification finale du nombre de colonnes
            if df.shape[1] != len(self.fixed_features):
                raise ValueError(f"Incohérence dans le nombre de colonnes: {df.shape[1]} vs {len(self.fixed_features)} attendues")
                
            logger.info(f"DataFrame harmonisé selon la configuration précédente: {df.shape[1]} colonnes")
            return df
        
        # Déterminer le nombre cible de caractéristiques
        target_feature_count = force_feature_count
        if target_feature_count is None:
            if self.auto_optimize and self.optimal_feature_count is not None:
                target_feature_count = self.optimal_feature_count
                logger.info(f"Utilisation du nombre optimal de features déterminé par auto-optimisation: {target_feature_count}")
            else:
                target_feature_count = self.expected_feature_count
                logger.info(f"Utilisation du nombre de features par défaut: {target_feature_count}")

        # Optimisation du pipeline : s'il manque des features, générer via PCA, sinon sélectionner les plus pertinentes
        current_count = df.shape[1]
        if current_count < target_feature_count:
            missing = target_feature_count - current_count
            logger.info(f"Nombre de features obtenu ({current_count}) inférieur à {target_feature_count}; génération de {missing} features additionnelles via PCA.")
            try:
                from sklearn.decomposition import PCA
                pca = PCA(n_components=missing)
                extra_features = pca.fit_transform(df)
                extra_df = pd.DataFrame(extra_features, index=df.index,
                                        columns=[f"pca_feature_{i+1}" for i in range(missing)])
                df = pd.concat([df, extra_df], axis=1)
            except Exception as e:
                logger.warning(f"Erreur lors de la génération des composantes PCA: {str(e)}. Ajout de colonnes de zéros.")
                for i in range(missing):
                    df[f"extra_feature_{i+1}"] = 0
        elif current_count > target_feature_count:
            logger.info(f"Nombre de features obtenu ({current_count}) supérieur à {target_feature_count}; sélection des features.")
            
            # Sélection basée sur les importances de caractéristiques si disponibles
            if self.feature_importances and enforce_consistency:
                sorted_features = sorted(self.feature_importances.items(), key=lambda x: x[1], reverse=True)
                top_features = [f for f, _ in sorted_features[:target_feature_count] if f in df.columns]
                
                # S'assurer d'avoir assez de features
                if len(top_features) < target_feature_count:
                    remaining_features = [f for f in df.columns if f not in top_features]
                    top_features.extend(remaining_features[:target_feature_count - len(top_features)])
                
                df = df[top_features[:target_feature_count]]
                logger.info("Sélection basée sur l'importance des caractéristiques.")
            else:
                # Sélection basée sur la variance
                variances = df.var()
                selected_columns = variances.sort_values(ascending=False).head(target_feature_count).index.tolist()
                df = df[selected_columns]
                logger.info("Sélection basée sur la variance des caractéristiques.")
        
        # Stocker la liste de features finales pour assurer la cohérence
        if enforce_consistency:
            self.fixed_features = df.columns.tolist()

        if enforce_consistency and force_feature_count == 66:
            logger.info("Forcing features to 66 for consistent sequence dimensions.")

        # Vérification finale du nombre de colonnes
        if df.shape[1] != target_feature_count:
            raise ValueError(f"Expected {target_feature_count} features, got {df.shape[1]}")

        logger.info(f"Nombre de features utilisé pour l'entraînement: {df.shape[1]}")
        
        # Ajouter une vérification supplémentaire après l'harmonisation
        if enforce_consistency:
            # Enregistrer la configuration pour utilisation future
            self._save_feature_metadata(df)

        # Ensure these columns are included in the final DataFrame
        df = df.reindex(columns=df.columns.union([
            "ema_9", "ema_21", "ema_50", "ema_200",
            "dist_to_ema_9", "dist_to_ema_21", "dist_to_ema_50", "dist_to_ema_200"
        ]), fill_value=0)

        return df

    def _save_feature_metadata(self, df: pd.DataFrame) -> None:
        """
        Enregistre les métadonnées des caractéristiques pour assurer la cohérence
        
        Args:
            df: DataFrame avec les caractéristiques finalisées
        """
        # Enregistrer la liste exacte des colonnes
        self.fixed_features = df.columns.tolist()
        
        # Calculer des statistiques de base sur les colonnes
        feature_stats = {}
        for col in df.columns:
            try:
                feature_stats[col] = {
                    "mean": float(df[col].mean()),
                    "std": float(df[col].std()),
                    "min": float(df[col].min()),
                    "max": float(df[col].max()),
                    "nan_count": int(df[col].isna().sum())
                }
            except Exception:
                # En cas d'erreur, enregistrer des informations minimales
                feature_stats[col] = {"calculable": False}
        
        # Sauvegarder les métadonnées complètes
        metadata = {
            "timestamp": datetime.now().isoformat(),
            "feature_count": len(self.fixed_features),
            "feature_list": self.fixed_features,
            "feature_stats": feature_stats
        }
        
        # Enregistrer les métadonnées dans un fichier séparé pour référence
        try:
            metadata_path = os.path.join(self.scalers_path, "feature_metadata.json")
            os.makedirs(os.path.dirname(metadata_path), exist_ok=True)
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2, default=str)
            logger.info(f"Métadonnées des caractéristiques enregistrées: {metadata_path}")
        except Exception as e:
            logger.warning(f"Impossible d'enregistrer les métadonnées des caractéristiques: {str(e)}")
    
    def optimize_feature_count(self, data: pd.DataFrame, target_variable: str = None, 
                              min_features: int = 20, max_features: int = 100, 
                              step_size: int = 10, cv_folds: int = 3) -> int:
        """
        Détermine le nombre optimal de caractéristiques en évaluant différentes configurations
        
        Args:
            data: DataFrame avec données OHLCV
            target_variable: Variable cible à prédire (par défaut: crée une direction de prix)
            min_features: Nombre minimum de caractéristiques à considérer
            max_features: Nombre maximum de caractéristiques à considérer
            step_size: Incrément pour tester différents nombres de caractéristiques
            cv_folds: Nombre de plis pour la validation croisée
            
        Returns:
            Nombre optimal de caractéristiques basé sur les métriques de performance
        """
        # Générer toutes les caractéristiques d'abord
        df_with_features = self.create_features(data, enforce_consistency=False)
        
        # Créer une variable cible simple si non fournie (direction du prix)
        if target_variable is None or target_variable not in df_with_features.columns:
            target_variable = 'price_direction'
            df_with_features[target_variable] = (df_with_features['close'].shift(-1) > df_with_features['close']).astype(int)
        
        # Éliminer les lignes avec valeurs manquantes dans la variable cible
        df_with_features = df_with_features.dropna(subset=[target_variable])
        
        # Initialiser le suivi des résultats
        results = {}
        feature_importances = {}
        
        # Ajuster la plage de comptage de features selon les données disponibles
        max_possible_features = min(max_features, df_with_features.shape[1] - 1)  # -1 pour exclure la cible
        feature_counts = range(min_features, max_possible_features + 1, step_size)
        
        # Validation croisée temporelle pour les séries chronologiques
        tscv = TimeSeriesSplit(n_splits=cv_folds)
        
        # Pour chaque nombre de caractéristiques
        for n_features in feature_counts:
            logger.info(f"Évaluation des performances avec {n_features} caractéristiques")
            
            # Sélectionner les caractéristiques par variance pour ce premier passage
            feature_variances = df_with_features.drop(columns=[target_variable]).var()
            selected_features = feature_variances.nlargest(n_features).index.tolist()
            
            # Préparer les données pour la validation croisée
            X = df_with_features[selected_features]
            y = df_with_features[target_variable]
            
            # Effectuer la validation croisée temporelle
            model = RandomForestClassifier(n_estimators=50, random_state=42, n_jobs=-1)
            
            # Métriques pour chaque pli
            fold_scores = {
                'accuracy': [],
                'f1': [],
                'precision': [],
                'recall': []
            }
            
            # Validation croisée manuelle pour pouvoir obtenir l'importance des features
            for train_idx, test_idx in tscv.split(X):
                X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
                y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
                
                # Entraîner le modèle
                model.fit(X_train, y_train)
                
                # Prédictions
                y_pred = model.predict(X_test)
                
                # Calculer les métriques
                fold_scores['accuracy'].append(accuracy_score(y_test, y_pred))
                fold_scores['f1'].append(f1_score(y_test, y_pred, average='weighted'))
                fold_scores['precision'].append(precision_score(y_test, y_pred, average='weighted', zero_division=0))
                fold_scores['recall'].append(recall_score(y_test, y_pred, average='weighted', zero_division=0))
            
            # Calculer les moyennes des métriques
            avg_scores = {k: np.mean(v) for k, v in fold_scores.items()}
            results[n_features] = avg_scores
            
            # Réentraîner sur toutes les données pour obtenir l'importance globale des features
            model.fit(X, y)
            
            # Stocker l'importance des features
            feature_importance = dict(zip(selected_features, model.feature_importances_))
            
            # Mettre à jour le dictionnaire global d'importance des features
            for feature, importance in feature_importance.items():
                if feature in feature_importances:
                    feature_importances[feature] = max(feature_importances[feature], importance)
                else:
                    feature_importances[feature] = importance
            
            logger.info(f"Performance moyenne avec {n_features} features: F1={avg_scores['f1']:.4f}, "
                       f"Accuracy={avg_scores['accuracy']:.4f}, Precision={avg_scores['precision']:.4f}, "
                       f"Recall={avg_scores['recall']:.4f}")
        
        # Trouver le nombre optimal de caractéristiques (meilleur F1-score)
        optimal_count = max(results.items(), key=lambda x: x[1]['f1'])[0]
        best_f1 = results[optimal_count]['f1']
        
        # Vérifier si la configuration actuelle (expected_feature_count) est proche de l'optimal
        current_performance = next((results[count]['f1'] for count in results if count == self.expected_feature_count), None)
        performance_threshold = 0.95  # 95% de la performance optimale
        
        if current_performance and current_performance >= best_f1 * performance_threshold:
            logger.info(f"La configuration actuelle ({self.expected_feature_count} features) avec F1={current_performance:.4f} "
                      f"est suffisamment proche de l'optimal ({optimal_count} features) avec F1={best_f1:.4f}. "
                      f"Conservation du nombre actuel de features.")
            optimal_count = self.expected_feature_count
        
        # Stocker les importances des features pour une utilisation future
        self.feature_importances = feature_importances
        
        # Stocker le nombre optimal de caractéristiques et mettre à jour la configuration centralisée
        self.optimal_feature_count = optimal_count
        update_optimal_feature_count(optimal_count)
        
        # Rapport des résultats
        logger.info(f"Résultats de l'évaluation du nombre de caractéristiques:")
        for count, metrics in sorted(results.items()):
            logger.info(f"  {count} caractéristiques: F1={metrics['f1']:.4f}, Accuracy={metrics['accuracy']:.4f}")
        logger.info(f"Nombre optimal de caractéristiques: {optimal_count} avec F1={results[optimal_count]['f1']:.4f}")
        
        return optimal_count

    def optimize_and_configure(self, data: pd.DataFrame, save_config: bool = True) -> Dict:
        """
        Effectue l'optimisation complète des caractéristiques et configure le pipeline
        
        Args:
            data: DataFrame avec données OHLCV
            save_config: Sauvegarder les résultats d'optimisation sur disque
            
        Returns:
            Dictionnaire avec les résultats d'optimisation
        """
        # Forcer la ré-optimisation, ignorer self.optimal_feature_count
        optimal_count = self.optimize_feature_count(data)
        
        # Mettre à jour les paramètres de la classe et la configuration centralisée
        self.expected_feature_count = optimal_count
        self.optimal_feature_count = optimal_count
        
        # Dans les tests, éviter de mettre à jour la config centralisée pour ne pas perturber d'autres tests
        if not 'unittest' in sys.modules:
            update_optimal_feature_count(optimal_count)
        
        # Générer l'ensemble final de caractéristiques avec le nombre optimal - forcer ce nombre
        optimized_features = self.create_features(
            data, 
            enforce_consistency=True,
            force_feature_count=optimal_count
        )
        
        # Sauvegarder la configuration si demandé
        if save_config:
            config_path = os.path.join(self.scalers_path, "feature_config.json")
            
            config = {
                "optimal_feature_count": optimal_count,
                "feature_list": self.fixed_features,
                "timestamp": datetime.now().isoformat(),
                "data_rows": len(data),
                "top_features": sorted([(f, i) for f, i in self.feature_importances.items()], 
                                      key=lambda x: x[1], reverse=True)[:20]
            }
            
            os.makedirs(os.path.dirname(config_path), exist_ok=True)
            with open(config_path, 'w') as f:
                json.dump(config, f, indent=2, default=str)
            
            logger.info(f"Configuration des caractéristiques sauvegardée dans {config_path}")
        
        # Retourner les résultats
        return {
            "optimal_feature_count": optimal_count,
            "features": self.fixed_features,
            "feature_importances": sorted(self.feature_importances.items(), key=lambda x: x[1], reverse=True),
            "feature_dataframe": optimized_features
        }

    def load_feature_configuration(self) -> bool:
        """
        Charge la configuration optimisée des caractéristiques si elle existe
        
        Returns:
            True si la configuration a été chargée avec succès, False sinon
        """
        # D'abord essayer de charger la configuration standard
        config_path = os.path.join(self.scalers_path, "feature_config.json")
        metadata_path = os.path.join(self.scalers_path, "feature_metadata.json")
        
        # Vérifier si les fichiers existent
        config_exists = os.path.exists(config_path)
        metadata_exists = os.path.exists(metadata_path)
        
        # Si aucun fichier local n'existe, utiliser la configuration centralisée
        if not config_exists and not metadata_exists:
            # Charger depuis la configuration centralisée si aucune valeur explicite n'a été fournie
            if self.optimal_feature_count is None:
                centralized_feature_count = get_optimal_feature_count()
                self.optimal_feature_count = centralized_feature_count
                
                logger.info(f"Configuration locale non trouvée, utilisation de la configuration centralisée: {centralized_feature_count} features")
            else:
                logger.info(f"Configuration locale non trouvée, conservation de la valeur explicite: {self.optimal_feature_count} features")
                
            return True
        
        try:
            # Charger les métadonnées si disponibles (plus détaillées)
            if metadata_exists:
                with open(metadata_path, 'r') as f:
                    metadata = json.load(f)
                
                self.fixed_features = metadata.get("feature_list")
                # Ne pas écraser la valeur explicite si elle existe déjà
                if self.optimal_feature_count is None and "feature_count" in metadata:
                    self.optimal_feature_count = metadata.get("feature_count")
                    
                logger.info(f"Métadonnées des caractéristiques chargées: {metadata_path}")
                
                # Garder une référence aux statistiques pour validation
                self.feature_stats = metadata.get("feature_stats", {})
            
            # Sinon, essayer la configuration standard
            elif config_exists:
                with open(config_path, 'r') as f:
                    config = json.load(f)
                
                # Ne pas écraser une valeur explicite si elle existe déjà
                if self.optimal_feature_count is None and "optimal_feature_count" in config:
                    self.optimal_feature_count = config.get("optimal_feature_count")
                    
                self.fixed_features = config.get("feature_list")
                
                if "top_features" in config:
                    self.feature_importances = {f: i for f, i in config["top_features"]}
                
                logger.info(f"Configuration des caractéristiques chargée: {config_path}")
            
            if self.fixed_features:
                logger.info(f"Liste de caractéristiques chargée: {len(self.fixed_features)} colonnes")
                
                # Si optimal_feature_count n'est toujours pas défini, utiliser la longueur des fixed_features
                if self.optimal_feature_count is None:
                    self.optimal_feature_count = len(self.fixed_features)
                
                # Dans les tests, on veut que la valeur explicite soit prioritaire
                # Synchroniser avec la configuration centralisée seulement si nécessaire
                # et si on n'a pas déjà une valeur explicite
                if self.optimal_feature_count is not None:
                    # Mettre à jour la config centralisée seulement si c'est un cas légitime de production
                    # pas dans les tests
                    if not 'unittest' in sys.modules:
                        update_optimal_feature_count(self.optimal_feature_count)
                        logger.info(f"Configuration centralisée mise à jour avec {self.optimal_feature_count} features")
                
                return True
            else:
                logger.warning("Liste de caractéristiques vide ou non valide dans la configuration")
                return False
                
        except Exception as e:
            logger.error(f"Erreur lors du chargement de la configuration des caractéristiques: {str(e)}")
            return False

    def scale_features(self, data: pd.DataFrame, is_training: bool = True,
                     method: str = 'standard', feature_group: str = 'lstm') -> pd.DataFrame:
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
            Tuple (X, y_list) pour l'entraînement ou (X, None) pour la prédiction
        """
        # Force columns to match FEATURE_COLUMNS
        data = data.reindex(columns=FEATURE_COLUMNS, fill_value=0).copy()
        if data.shape[1] != len(FEATURE_COLUMNS):
            logger.warning(f"Feature mismatch: expected {len(FEATURE_COLUMNS)}, found {data.shape[1]}")
        
        # Sélectionner les colonnes de caractéristiques
        feature_cols = data.select_dtypes(include=['float64', 'int64']).columns.tolist()
        cols_to_exclude = ['timestamp', 'date']
        feature_cols = [col for col in feature_cols if col not in cols_to_exclude]
        
        if is_training:
            # Use the maximum horizon to ensure alignment across toutes les sorties
            max_horizon = max(horizons)
            num_samples = len(data) - sequence_length - max_horizon
        else:
            num_samples = len(data) - sequence_length
        
        # Créer les séquences d'entrée avec num_samples
        X = []
        for i in range(num_samples):
            X.append(data[feature_cols].iloc[i:i+sequence_length].values.astype(np.float32))
        X = np.array(X)
        
        if not is_training:
            # Instead of just returning X, return a tuple (X, None) for consistency
            logger.debug(f"Returning prediction data without labels: X shape {X.shape}")
            return X, None
        
        # Créer les labels pour chaque horizon avec the same num_samples
        y_list = []
        for horizon in horizons:
            y_direction = []
            y_volatility = []
            y_volume = []
            y_momentum = []
            for i in range(num_samples):
                current_price = data['close'].iloc[i+sequence_length-1]
                future_price = data['close'].iloc[i+sequence_length+horizon-1]
                direction = 1 if future_price > current_price else 0
                y_direction.append(direction)
                future_returns = data['close'].iloc[i+sequence_length:i+sequence_length+horizon].pct_change().dropna()
                volatility = future_returns.std() * np.sqrt(horizon)
                y_volatility.append(volatility)
                current_volume = data['volume'].iloc[i+sequence_length-1]
                future_volume = data['volume'].iloc[i+sequence_length:i+sequence_length+horizon].mean()
                relative_volume = future_volume / current_volume if current_volume > 0 else 1.0
                y_volume.append(relative_volume)
                price_change_pct = (future_price - current_price) / current_price
                momentum = np.tanh(price_change_pct * 5)
                y_momentum.append(momentum)
            
            y_list.extend([np.array(y_direction), np.array(y_volatility), np.array(y_volume), np.array(y_momentum)])
        
        # Before returning, ensure X has the right dimensions (3D)
        if len(X.shape) == 2:
            # If X is 2D (n_samples, n_features), reshape to 3D
            # This can happen with some data preparation methods
            n_samples, n_features = X.shape
            X = X.reshape(n_samples, 1, n_features)
            logger.debug(f"Reshaped X from 2D to 3D: {X.shape}")
        
        # Validate dimensions before return
        if len(X.shape) != 3:
            error_msg = f"Expected X to be 3D, but got shape {X.shape} with dimensions {len(X.shape)}"
            logger.error(error_msg)
            raise ValueError(error_msg)
        
        # Verify we have data to return
        if len(X) == 0:
            logger.warning("No sequences were generated - input data may be too short")
            
        # Log the return shape
        logger.debug(f"Returning training data: X shape {X.shape}, y_list length {len(y_list)}")
        return X, y_list

    def evaluate_feature_impact(self, data: pd.DataFrame, target_variable: str = None) -> pd.DataFrame:
        """
        Évalue l'impact de chaque caractéristique sur la variable cible
        
        Args:
            data: DataFrame avec les données et caractéristiques
            target_variable: Variable cible à prédire (par défaut: crée une direction de prix)
            
        Returns:
            DataFrame avec les métriques d'impact de chaque caractéristique
        """
        # Générer les caractéristiques
        df_features = self.create_features(data)
        
        # Créer une variable cible si non fournie
        if target_variable is None or target_variable not in df_features.columns:
            target_variable = 'price_direction'
            df_features[target_variable] = (df_features['close'].shift(-1) > df_features['close']).astype(int)
        
        # Éliminer les lignes avec valeurs manquantes
        df_features = df_features.dropna()
        
        # Initialiser le dictionnaire pour les métriques d'impact
        impact_metrics = {}
        
        # 1. Analyse de corrélation
        correlations = df_features.corr()[target_variable].drop(target_variable)
        
        # 2. Importance des caractéristiques via Random Forest
        from sklearn.ensemble import RandomForestClassifier
        
        X = df_features.drop(columns=[target_variable])
        y = df_features[target_variable]
        
        # Entraîner un modèle simple pour l'importance des caractéristiques
        model = RandomForestClassifier(n_estimators=50, random_state=42)
        model.fit(X, y)
        
        # Récupérer l'importance des caractéristiques
        importances = model.feature_importances_
        
        # 3. Résultat combiné
        feature_impact = pd.DataFrame({
            'feature': X.columns,
            'correlation': correlations.values,
            'abs_correlation': correlations.abs().values,
            'importance': importances
        })
        
        # Trier par importance
        feature_impact = feature_impact.sort_values('importance', ascending=False)
        
        # Calculer un score combiné
        feature_impact['combined_score'] = (
            0.4 * feature_impact['abs_correlation'] / feature_impact['abs_correlation'].max() +
            0.6 * feature_impact['importance'] / feature_impact['importance'].max()
        )
        
        # Stocker les résultats pour utilisation future
        self.feature_impact = feature_impact
        
        # Mettre à jour les importances de caractéristiques
        self.feature_importances = dict(zip(feature_impact['feature'], feature_impact['importance']))
        
        return feature_impact
    
    def verify_feature_consistency(self, data: pd.DataFrame) -> Dict:
        """
        Vérifie la cohérence des caractéristiques générées par rapport à la configuration enregistrée
        
        Args:
            data: DataFrame avec les données brutes
            
        Returns:
            Dictionnaire avec les résultats de la vérification
        """
        # Vérifier si une configuration existe
        if self.fixed_features is None:
            return {"consistent": False, "reason": "Pas de configuration de référence"}
        
        # Générer les caractéristiques
        try:
            df_features = self.create_features(data, enforce_consistency=True)
            
            # Vérifier le nombre de colonnes
            expected_count = len(self.fixed_features)
            actual_count = df_features.shape[1]
            columns_match = (expected_count == actual_count)
            
            # Vérifier l'ordre des colonnes
            columns_order_match = all(df_features.columns[i] == self.fixed_features[i] 
                                    for i in range(min(len(df_features.columns), len(self.fixed_features))))
            
            # Vérifier la présence de valeurs aberrantes si des statistiques sont disponibles
            outliers_check = {}
            if hasattr(self, 'feature_stats'):
                for col in df_features.columns:
                    if col in self.feature_stats and "std" in self.feature_stats[col]:
                        ref_mean = self.feature_stats[col]["mean"]
                        ref_std = self.feature_stats[col]["std"]
                        current_mean = df_features[col].mean()
                        
                        # Vérifier si la moyenne actuelle est très différente de la référence
                        if abs(current_mean - ref_mean) > 3 * ref_std:
                            outliers_check[col] = {
                                "reference_mean": ref_mean,
                                "current_mean": float(current_mean),
                                "deviation_sigmas": float(abs(current_mean - ref_mean) / ref_std)
                            }
            
            return {
                "consistent": columns_match and columns_order_match and not outliers_check,
                "columns_count_match": columns_match,
                "expected_columns": expected_count,
                "actual_columns": actual_count,
                "columns_order_match": columns_order_match,
                "outliers_detected": len(outliers_check) > 0,
                "outliers": outliers_check
            }
            
        except Exception as e:
            return {"consistent": False, "reason": str(e)}

def add_market_sentiment_features(df: pd.DataFrame) -> pd.DataFrame:
    # Calculer l'indice de force relative (RSI) sur plusieurs périodes et calculer la pente sur 5 périodes
    for period in [7, 14, 21]:
        df[f'rsi_{period}_slope'] = talib.RSI(df['close'], timeperiod=period).pct_change(5)
    
    # Calculer la divergence entre les variations de prix et de volume
    df['price_volume_divergence'] = df['close'].pct_change() * df['volume'].pct_change()
    
    # Si les indicateurs MACD sont présents, créer une caractéristique de crossover
    # Si les indicateurs MACD sont présents, créer une caractéristique de crossover
    if 'macd' in df.columns and 'macd_signal' in df.columns:
        df['macd_crossover'] = (df['macd'] - df['macd_signal']).apply(lambda x: 1 if x > 0 else -1)
    return df