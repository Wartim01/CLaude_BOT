"""
Module avancé pour la détection d'anomalies et d'événements extrêmes (black swan) sur les marchés financiers
Intègre plusieurs approches statistiques et algorithmiques pour identifier les conditions de marché anormales
"""
import numpy as np
import pandas as pd
from typing import Dict, List, Union, Optional
from scipy import stats
from statsmodels.tsa.stattools import adfuller
import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, LSTM, Dropout, Input
import warnings
import os
import pickle
from datetime import datetime, timedelta

from utils.logger import setup_logger

logger = setup_logger("market_anomaly_detector")

class MarketAnomalyDetector:
    """
    Détecteur d'anomalies et d'événements extrêmes pour les marchés financiers
    Utilise plusieurs méthodes complémentaires:
    1. Tests statistiques (détection d'outliers, fat tails, rupture de stationnarité)
    2. Analyse des microstructures de marché (rupture de l'algorithme de matching, flash crashes)
    3. Détection d'anomalies basée sur l'apprentissage automatique (isolation forest, autoencoder)
    """
    def __init__(self, 
                lookback_period: int = 100,
                confidence_level: float = 0.99,
                volatility_threshold: float = 3.0,
                volume_threshold: float = 5.0,
                price_gap_threshold: float = 3.0,
                use_ml_models: bool = True,
                model_dir: Optional[str] = None):
        """
        Initialise le détecteur d'anomalies
        
        Args:
            lookback_period: Période d'historique pour les calculs statistiques
            confidence_level: Niveau de confiance pour la détection des anomalies
            volatility_threshold: Seuil de multiplication d'ATR pour la volatilité extrême
            volume_threshold: Seuil de multiplication du volume moyen pour volume extrême
            price_gap_threshold: Seuil de multiplication de l'ATR pour les gaps de prix
            use_ml_models: Utiliser des modèles de ML pour la détection d'anomalies
            model_dir: Répertoire pour sauvegarder/charger les modèles
        """
        self.lookback_period = lookback_period
        self.confidence_level = confidence_level
        self.volatility_threshold = volatility_threshold
        self.volume_threshold = volume_threshold
        self.price_gap_threshold = price_gap_threshold
        self.use_ml_models = use_ml_models
        self.model_dir = model_dir
        
        # Historique des anomalies détectées
        self.anomaly_history = []
        
        # Initialisation des modèles de ML pour la détection d'anomalies
        self.isolation_forest = None
        self.autoencoder = None
        
        # Charger les modèles si disponibles
        if use_ml_models and model_dir:
            self._load_models()
    
    def detect_anomalies(self, data: pd.DataFrame, current_price: float = None,
                       return_details: bool = False) -> Union[Dict, bool]:
        """
        Détecte les anomalies dans les données de marché fournies
        
        Args:
            data: DataFrame avec les données OHLCV
            current_price: Prix actuel (si différent du dernier prix dans data)
            return_details: Retourner les détails de l'analyse
            
        Returns:
            True/False ou dictionnaire détaillé si anomalie détectée
        """
        if len(data) < self.lookback_period:
            logger.warning(f"Données insuffisantes pour la détection d'anomalies: {len(data)} < {self.lookback_period}")
            return False if not return_details else {"detected": False, "reason": "Données insuffisantes"}
        
        # Utiliser les données récentes pour l'analyse
        recent_data = data.tail(self.lookback_period).copy()
        
        # Mise à jour du prix actuel si fourni
        if current_price is not None:
            current_close = current_price
        else:
            current_close = recent_data['close'].iloc[-1]
        
        # Résultats de détection pour différentes méthodes
        results = {}
        
        # 1. Vérifier la volatilité extrême
        volatility_anomaly = self._detect_volatility_anomaly(recent_data, current_close)
        results["volatility_anomaly"] = volatility_anomaly
        
        # 2. Vérifier les gaps de prix significatifs
        price_gap_anomaly = self._detect_price_gap(recent_data, current_close)
        results["price_gap_anomaly"] = price_gap_anomaly
        
        # 3. Vérifier le volume anormal
        volume_anomaly = self._detect_volume_anomaly(recent_data)
        results["volume_anomaly"] = volume_anomaly
        
        # 4. Vérifier les fat tails dans la distribution des rendements
        fat_tails = self._detect_fat_tails(recent_data)
        results["fat_tails"] = fat_tails
        
        # 5. Vérifier la rupture de stationnarité
        stationarity_break = self._detect_stationarity_break(recent_data)
        results["stationarity_break"] = stationarity_break
        
        # 6. Vérifier les anomalies de microstructure
        microstructure_anomaly = self._detect_microstructure_anomaly(recent_data, current_close)
        results["microstructure_anomaly"] = microstructure_anomaly
        
        # 7. Utiliser les modèles de ML pour la détection d'anomalies
        ml_anomaly = self._detect_ml_anomalies(recent_data) if self.use_ml_models else False
        results["ml_anomaly"] = ml_anomaly
        
        # 8. Vérifier le momentum extrême
        momentum_anomaly = self._detect_momentum_anomaly(recent_data)
        results["momentum_anomaly"] = momentum_anomaly
        
        # Combinaison des résultats
        # Une anomalie est détectée si au moins deux méthodes différentes signalent une anomalie
        anomaly_count = sum(1 for result in results.values() if result["detected"])
        anomaly_detected = anomaly_count >= 2  # Au moins deux méthodes doivent détecter une anomalie
        
        # Déterminer la raison principale
        primary_reason = None
        if anomaly_detected:
            # Trouver la méthode avec le score d'anomalie le plus élevé
            max_score = 0
            for method, result in results.items():
                if result["detected"] and result.get("score", 0) > max_score:
                    max_score = result.get("score", 0)
                    primary_reason = result.get("reason", method)
        
        # Créer le résultat final
        result = {
            "detected": anomaly_detected,
            "reason": primary_reason if anomaly_detected else None,
            "anomaly_count": anomaly_count,
            "timestamp": datetime.now().isoformat(),
            "symbol": data.get('symbol', 'unknown'),
            "current_price": current_close
        }
        
        # Ajouter les détails si demandé
        if return_details:
            result["details"] = results
        
        # Enregistrer l'anomalie dans l'historique si détectée
        if anomaly_detected:
            self.anomaly_history.append(result)
            logger.warning(f"Anomalie de marché détectée: {primary_reason}")
        
        return result
    
    def _detect_volatility_anomaly(self, data: pd.DataFrame, current_price: float) -> Dict:
        """
        Détecte une volatilité anormalement élevée
        
        Args:
            data: DataFrame avec les données OHLCV
            current_price: Prix actuel
            
        Returns:
            Résultat de la détection
        """
        # Calculer l'ATR sur la période de lookback
        try:
            # Calculer le True Range
            high_low = data['high'] - data['low']
            high_close = np.abs(data['high'] - data['close'].shift(1))
            low_close = np.abs(data['low'] - data['close'].shift(1))
            
            ranges = pd.concat([high_low, high_close, low_close], axis=1)
            true_range = np.max(ranges, axis=1)
            
            # Calculer l'ATR (Average True Range)
            atr = true_range.rolling(window=14).mean().iloc[-1]
            
            # Calculer la volatilité récente (écart-type des rendements)
            returns = data['close'].pct_change().dropna()
            recent_volatility = returns.tail(10).std() * np.sqrt(10)  # Annualisé à 10 périodes
            
            # Calculer la volatilité historique
            historical_volatility = returns.std() * np.sqrt(self.lookback_period)
            
            # Calculer la volatilité relative
            volatility_ratio = recent_volatility / historical_volatility if historical_volatility > 0 else 1.0
            
            # Vérifier si la volatilité actuelle dépasse le seuil
            is_anomaly = volatility_ratio > self.volatility_threshold
            
            # Calculer un score d'anomalie
            score = volatility_ratio / self.volatility_threshold
            
            return {
                "detected": is_anomaly,
                "reason": f"Volatilité extrême ({volatility_ratio:.2f}x la normale)" if is_anomaly else None,
                "atr": atr,
                "recent_volatility": recent_volatility,
                "historical_volatility": historical_volatility,
                "volatility_ratio": volatility_ratio,
                "score": score
            }
        
        except Exception as e:
            logger.error(f"Erreur lors de la détection d'anomalie de volatilité: {str(e)}")
            return {"detected": False, "reason": f"Erreur: {str(e)}", "score": 0}
    
    def _detect_price_gap(self, data: pd.DataFrame, current_price: float) -> Dict:
        """
        Détecte un gap de prix significatif
        
        Args:
            data: DataFrame avec les données OHLCV
            current_price: Prix actuel
            
        Returns:
            Résultat de la détection
        """
        try:
            # Calculer l'ATR pour normaliser les gaps
            high_low = data['high'] - data['low']
            high_close = np.abs(data['high'] - data['close'].shift(1))
            low_close = np.abs(data['low'] - data['close'].shift(1))
            
            ranges = pd.concat([high_low, high_close, low_close], axis=1)
            true_range = np.max(ranges, axis=1)
            
            atr = true_range.rolling(window=14).mean().iloc[-1]
            
            # Calculer le gap entre le prix actuel et le précédent
            previous_close = data['close'].iloc[-2]
            gap_size = abs(current_price - previous_close)
            
            # Normaliser par l'ATR
            normalized_gap = gap_size / atr if atr > 0 else 0
            
            # Vérifier si le gap dépasse le seuil
            is_anomaly = normalized_gap > self.price_gap_threshold
            
            # Calculer un score d'anomalie
            score = normalized_gap / self.price_gap_threshold
            
            return {
                "detected": is_anomaly,
                "reason": f"Gap de prix significatif ({normalized_gap:.2f}x ATR)" if is_anomaly else None,
                "gap_size": gap_size,
                "normalized_gap": normalized_gap,
                "atr": atr,
                "score": score
            }
        
        except Exception as e:
            logger.error(f"Erreur lors de la détection de gap de prix: {str(e)}")
            return {"detected": False, "reason": f"Erreur: {str(e)}", "score": 0}
    
    def _detect_volume_anomaly(self, data: pd.DataFrame) -> Dict:
        """
        Détecte un volume anormal
        
        Args:
            data: DataFrame avec les données OHLCV
            
        Returns:
            Résultat de la détection
        """
        try:
            # Calculer le volume moyen
            avg_volume = data['volume'].mean()
            
            # Vérifier le volume récent
            recent_volume = data['volume'].iloc[-1]
            
            # Calculer le ratio
            volume_ratio = recent_volume / avg_volume if avg_volume > 0 else 1.0
            
            # Vérifier si le volume dépasse le seuil
            is_anomaly = volume_ratio > self.volume_threshold
            
            # Calculer un score d'anomalie
            score = volume_ratio / self.volume_threshold
            
            return {
                "detected": is_anomaly,
                "reason": f"Volume anormalement élevé ({volume_ratio:.2f}x la moyenne)" if is_anomaly else None,
                "recent_volume": recent_volume,
                "avg_volume": avg_volume,
                "volume_ratio": volume_ratio,
                "score": score
            }
        
        except Exception as e:
            logger.error(f"Erreur lors de la détection d'anomalie de volume: {str(e)}")
            return {"detected": False, "reason": f"Erreur: {str(e)}", "score": 0}
    
    def _detect_fat_tails(self, data: pd.DataFrame) -> Dict:
        """
        Détecte des queues de distribution épaisses (fat tails)
        
        Args:
            data: DataFrame avec les données OHLCV
            
        Returns:
            Résultat de la détection
        """
        try:
            # Calculer les rendements
            returns = data['close'].pct_change().dropna()
            
            # Ignorer si pas assez de données
            if len(returns) < 30:
                return {"detected": False, "reason": "Données insuffisantes pour l'analyse des queues de distribution", "score": 0}
            
            # Calculer le kurtosis (mesure de l'épaisseur des queues)
            kurt = stats.kurtosis(returns)
            
            # Le kurtosis d'une distribution normale est de 0
            # Une valeur > 3 indique des queues épaisses
            is_anomaly = kurt > 3.0
            
            # Calculer un score d'anomalie
            score = kurt / 3.0 if kurt > 0 else 0
            
            return {
                "detected": is_anomaly,
                "reason": f"Distribution des rendements à queues épaisses (kurtosis={kurt:.2f})" if is_anomaly else None,
                "kurtosis": kurt,
                "score": score
            }
        
        except Exception as e:
            logger.error(f"Erreur lors de la détection des queues de distribution: {str(e)}")
            return {"detected": False, "reason": f"Erreur: {str(e)}", "score": 0}
    
    def _detect_stationarity_break(self, data: pd.DataFrame) -> Dict:
        """
        Détecte une rupture de stationnarité dans la série
        
        Args:
            data: DataFrame avec les données OHLCV
            
        Returns:
            Résultat de la détection
        """
        try:
            # Calculer les rendements
            returns = data['close'].pct_change().dropna()
            
            # Ignorer si pas assez de données
            if len(returns) < 30:
                return {"detected": False, "reason": "Données insuffisantes pour le test de stationnarité", "score": 0}
            
            # Test de Dickey-Fuller augmenté
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                result = adfuller(returns)
            
            # Extraire les statistiques
            adf_stat = result[0]
            p_value = result[1]
            
            # Seuil de signification
            significance = 0.05
            
            # Si p-value > significance, alors la série n'est pas stationnaire
            is_anomaly = p_value > significance
            
            # Calculer un score d'anomalie
            score = p_value / significance if significance > 0 else 0
            
            return {
                "detected": is_anomaly,
                "reason": f"Rupture de stationnarité détectée (p-value={p_value:.4f})" if is_anomaly else None,
                "adf_statistic": adf_stat,
                "p_value": p_value,
                "score": score
            }
        
        except Exception as e:
            logger.error(f"Erreur lors du test de stationnarité: {str(e)}")
            return {"detected": False, "reason": f"Erreur: {str(e)}", "score": 0}
    
    def _detect_microstructure_anomaly(self, data: pd.DataFrame, current_price: float) -> Dict:
        """
        Détecte des anomalies dans la microstructure du marché
        
        Args:
            data: DataFrame avec les données OHLCV
            current_price: Prix actuel
            
        Returns:
            Résultat de la détection
        """
        try:
            # Calculer l'écart entre high/low et le prix de clôture
            recent_data = data.tail(5)  # 5 dernières périodes
            
            # Calculer les ratios high-close et low-close
            high_close_ratios = (recent_data['high'] - recent_data['close']) / recent_data['close']
            low_close_ratios = (recent_data['close'] - recent_data['low']) / recent_data['close']
            
            # Calculer les moyennes historiques
            hist_high_close_ratio = (data['high'] - data['close']) / data['close']
            hist_low_close_ratio = (data['close'] - data['low']) / data['close']
            
            avg_high_ratio = hist_high_close_ratio.mean()
            avg_low_ratio = hist_low_close_ratio.mean()
            
            # Vérifier si les ratios récents sont anormaux
            recent_high_ratio = high_close_ratios.mean()
            recent_low_ratio = low_close_ratios.mean()
            
            # Calculer les écarts
            high_deviation = recent_high_ratio / avg_high_ratio if avg_high_ratio > 0 else 1.0
            low_deviation = recent_low_ratio / avg_low_ratio if avg_low_ratio > 0 else 1.0
            
            # Vérifier si l'un des écarts dépasse le seuil
            high_anomaly = high_deviation > 2.0
            low_anomaly = low_deviation > 2.0
            
            is_anomaly = high_anomaly or low_anomaly
            
            # Calculer un score d'anomalie
            score = max(high_deviation, low_deviation) / 2.0
            
            # Construire la raison
            reason = None
            if is_anomaly:
                if high_anomaly and low_anomaly:
                    reason = f"Mèches anormales (H:{high_deviation:.2f}x, L:{low_deviation:.2f}x)"
                elif high_anomaly:
                    reason = f"Mèche supérieure anormale ({high_deviation:.2f}x la normale)"
                else:
                    reason = f"Mèche inférieure anormale ({low_deviation:.2f}x la normale)"
            
            return {
                "detected": is_anomaly,
                "reason": reason,
                "high_deviation": high_deviation,
                "low_deviation": low_deviation,
                "score": score
            }
        
        except Exception as e:
            logger.error(f"Erreur lors de la détection d'anomalie de microstructure: {str(e)}")
            return {"detected": False, "reason": f"Erreur: {str(e)}", "score": 0}
    
    def _detect_momentum_anomaly(self, data: pd.DataFrame) -> Dict:
        """
        Détecte un momentum extrême
        
        Args:
            data: DataFrame avec les données OHLCV
            
        Returns:
            Résultat de la détection
        """
        try:
            # Calculer les rendements sur plusieurs périodes
            returns_1d = data['close'].pct_change(1).iloc[-1]
            returns_3d = data['close'].pct_change(3).iloc[-1]
            returns_5d = data['close'].pct_change(5).iloc[-1]
            
            # Calculer les rendements historiques
            hist_returns_1d = data['close'].pct_change(1).std()
            hist_returns_3d = data['close'].pct_change(3).std()
            hist_returns_5d = data['close'].pct_change(5).std()
            
            # Calculer les z-scores
            z_score_1d = returns_1d / hist_returns_1d if hist_returns_1d > 0 else 0
            z_score_3d = returns_3d / hist_returns_3d if hist_returns_3d > 0 else 0
            z_score_5d = returns_5d / hist_returns_5d if hist_returns_5d > 0 else 0
            
            # Utiliser le z-score maximum
            max_z_score = max(abs(z_score_1d), abs(z_score_3d), abs(z_score_5d))
            
            # Seuil pour considérer un momentum comme anormal
            threshold = 2.5  # 2.5 écarts-types
            
            is_anomaly = max_z_score > threshold
            
            # Déterminer la direction du momentum
            direction = "haussier" if max(returns_1d, returns_3d, returns_5d) > 0 else "baissier"
            
            # Calculer un score d'anomalie
            score = max_z_score / threshold
            
            return {
                "detected": is_anomaly,
                "reason": f"Momentum {direction} extrême (z-score={max_z_score:.2f})" if is_anomaly else None,
                "z_score_1d": z_score_1d,
                "z_score_3d": z_score_3d,
                "z_score_5d": z_score_5d,
                "max_z_score": max_z_score,
                "direction": direction if is_anomaly else None,
                "score": score
            }
        
        except Exception as e:
            logger.error(f"Erreur lors de la détection d'anomalie de momentum: {str(e)}")
            return {"detected": False, "reason": f"Erreur: {str(e)}", "score": 0}
    
    def _detect_ml_anomalies(self, data: pd.DataFrame) -> Dict:
        """
        Utilise des modèles de ML pour détecter les anomalies
        
        Args:
            data: DataFrame avec les données OHLCV
            
        Returns:
            Résultat de la détection
        """
        # Si les modèles de ML ne sont pas activés
        if not self.use_ml_models:
            return {"detected": False, "reason": "Modèles ML non activés", "score": 0}
        
        try:
            # Extraire les caractéristiques pertinentes
            features = self._extract_anomaly_features(data)
            
            # Si pas de modèles chargés, les entraîner
            if self.isolation_forest is None or self.autoencoder is None:
                self._train_anomaly_models(data)
            
            # Prédictions de l'Isolation Forest (si disponible)
            forest_score = 0
            if self.isolation_forest is not None:
                # -1 pour les anomalies, 1 pour les normales, convertir à un score entre 0 et 1
                forest_pred = self.isolation_forest.predict([features])
                forest_score = self.isolation_forest.score_samples([features])[0]
                forest_score = 0.5 + (forest_score * -0.5)  # Convertir à un score d'anomalie (0-1)
            
            # Prédictions de l'Autoencoder (si disponible)
            autoencoder_score = 0
            if self.autoencoder is not None:
                # Préparation des données pour l'autoencoder
                ae_input = np.array([features])
                
                # Prédiction
                ae_pred = self.autoencoder.predict(ae_input)
                
                # Erreur de reconstruction
                reconstruction_error = np.mean(np.square(ae_input - ae_pred))
                
                # Normaliser l'erreur à un score entre 0 et 1
                autoencoder_score = min(1.0, reconstruction_error / 0.1)  # Seuil d'erreur de 0.1
            
            # Combiner les scores (moyenne)
            combined_score = (forest_score + autoencoder_score) / 2 if (self.isolation_forest is not None and self.autoencoder is not None) else max(forest_score, autoencoder_score)
            
            # Seuil pour considérer comme une anomalie
            threshold = 0.7
            is_anomaly = combined_score > threshold
            
            return {
                "detected": is_anomaly,
                "reason": f"Anomalie détectée par ML (score={combined_score:.2f})" if is_anomaly else None,
                "forest_score": forest_score,
                "autoencoder_score": autoencoder_score,
                "combined_score": combined_score,
                "score": combined_score
            }
        
        except Exception as e:
            logger.error(f"Erreur lors de la détection d'anomalies par ML: {str(e)}")
            return {"detected": False, "reason": f"Erreur: {str(e)}", "score": 0}
    
    def _extract_anomaly_features(self, data: pd.DataFrame) -> List[float]:
        """
        Extrait les caractéristiques pour la détection d'anomalies
        
        Args:
            data: DataFrame avec les données OHLCV
            
        Returns:
            Liste des caractéristiques
        """
        # Calculer les rendements
        returns = data['close'].pct_change().dropna()
        
        # Caractéristiques de volatilité
        volatility_1d = returns.tail(1).std()
        volatility_5d = returns.tail(5).std() * np.sqrt(5)
        volatility_10d = returns.tail(10).std() * np.sqrt(10)
        
        # Caractéristiques de momentum
        returns_1d = returns.iloc[-1]
        returns_3d = (data['close'].iloc[-1] / data['close'].iloc[-4] - 1) if len(data) >= 4 else 0
        returns_5d = (data['close'].iloc[-1] / data['close'].iloc[-6] - 1) if len(data) >= 6 else 0
        
        # Caractéristiques de volume
        volume_ratio_1d = data['volume'].iloc[-1] / data['volume'].iloc[-2] if data['volume'].iloc[-2] > 0 else 1
        volume_ratio_5d = data['volume'].iloc[-1] / data['volume'].tail(5).mean() if data['volume'].tail(5).mean() > 0 else 1
        
        # Caractéristiques des chandeliers
        body_size = abs(data['close'].iloc[-1] - data['open'].iloc[-1]) / data['open'].iloc[-1]
        upper_wick = (data['high'].iloc[-1] - max(data['open'].iloc[-1], data['close'].iloc[-1])) / data['open'].iloc[-1]
        lower_wick = (min(data['open'].iloc[-1], data['close'].iloc[-1]) - data['low'].iloc[-1]) / data['open'].iloc[-1]
        
        # Combiner les caractéristiques
        features = [
            volatility_1d, 
            volatility_5d, 
            volatility_10d,
            returns_1d, 
            returns_3d, 
            returns_5d,
            volume_ratio_1d, 
            volume_ratio_5d,
            body_size, 
            upper_wick, 
            lower_wick
        ]
        
        # Remplacer les valeurs NaN ou infinies
        features = [0.0 if (np.isnan(f) or np.isinf(f)) else f for f in features]
        
        return features
    
    def _train_anomaly_models(self, data: pd.DataFrame) -> None:
        """
        Entraîne les modèles de détection d'anomalies sur les données historiques
        
        Args:
            data: DataFrame avec les données OHLCV
        """
        from sklearn.ensemble import IsolationForest
        
        try:
            # Préparer les caractéristiques pour tous les points de données
            features_list = []
            
            # Fenêtre glissante pour extraire les caractéristiques
            for i in range(self.lookback_period, len(data)):
                window_data = data.iloc[i-self.lookback_period:i]
                features = self._extract_anomaly_features(window_data)
                features_list.append(features)
            
            # S'il n'y a pas assez de données, sortir
            if len(features_list) < 50:
                logger.warning("Données insuffisantes pour entraîner les modèles d'anomalies")
                return
            
            # Entraîner l'Isolation Forest
            self.isolation_forest = IsolationForest(
                n_estimators=100,
                max_samples='auto',
                contamination=0.05,  # 5% d'anomalies attendues
                random_state=42
            )
            self.isolation_forest.fit(features_list)
            
            # Entraîner l'Autoencoder
            self._train_autoencoder(np.array(features_list))
            
            # Sauvegarder les modèles si un répertoire est spécifié
            if self.model_dir:
                self._save_models()
                
            logger.info("Modèles de détection d'anomalies entraînés avec succès")
        
        except Exception as e:
            logger.error(f"Erreur lors de l'entraînement des modèles d'anomalies: {str(e)}")
    
    def _train_autoencoder(self, features: np.ndarray) -> None:
        """
        Entraîne un autoencoder pour la détection d'anomalies
        
        Args:
            features: Tableau numpy avec les caractéristiques
        """
        try:
            # Nombre de caractéristiques
            input_dim = features.shape[1]
            
            # Définir l'architecture de l'autoencoder
            input_layer = Input(shape=(input_dim,))
            
            # Encodeur
            encoded = Dense(8, activation='relu')(input_layer)
            encoded = Dense(4, activation='relu')(encoded)
            
            # Décodeur
            decoded = Dense(8, activation='relu')(encoded)
            decoded = Dense(input_dim, activation='linear')(decoded)
            
            # Modèle complet
            self.autoencoder = Model(input_layer, decoded)
            
            # Compilation
            self.autoencoder.compile(optimizer='adam', loss='mse')
            
            # Entraînement
            self.autoencoder.fit(
                features, 
                features,
                epochs=50,
                batch_size=32,
                shuffle=True,
                verbose=0
            )
            
            logger.info("Autoencoder entraîné avec succès")
        
        except Exception as e:
            logger.error(f"Erreur lors de l'entraînement de l'autoencoder: {str(e)}")
            self.autoencoder = None
    
    def _save_models(self) -> None:
        """Sauvegarde les modèles d'anomalies"""
        try:
            os.makedirs(self.model_dir, exist_ok=True)
            
            # Sauvegarder l'Isolation Forest
            if self.isolation_forest is not None:
                with open(os.path.join(self.model_dir, "isolation_forest.pkl"), 'wb') as f:
                    pickle.dump(self.isolation_forest, f)
            
            # Sauvegarder l'Autoencoder
            if self.autoencoder is not None:
                self.autoencoder.save(os.path.join(self.model_dir, "autoencoder"))
            
            logger.info(f"Modèles de détection d'anomalies sauvegardés dans {self.model_dir}")
        
        except Exception as e:
            logger.error(f"Erreur lors de la sauvegarde des modèles: {str(e)}")
    
    def _load_models(self) -> None:
        """Charge les modèles d'anomalies"""
        try:
            # Charger l'Isolation Forest
            forest_path = os.path.join(self.model_dir, "isolation_forest.pkl")
            if os.path.exists(forest_path):
                with open(forest_path, 'rb') as f:
                    self.isolation_forest = pickle.load(f)
                logger.info("Isolation Forest chargé")
            
            # Charger l'Autoencoder
            autoencoder_path = os.path.join(self.model_dir, "autoencoder")
            if os.path.exists(autoencoder_path):
                self.autoencoder = tf.keras.models.load_model(autoencoder_path)
                logger.info("Autoencoder chargé")
        
        except Exception as e:
            logger.error(f"Erreur lors du chargement des modèles: {str(e)}")
    
    def get_anomaly_history(self, limit: int = 10) -> List[Dict]:
        """
        Récupère l'historique des anomalies détectées
        
        Args:
            limit: Nombre maximum d'anomalies à retourner
            
        Returns:
            Liste des anomalies détectées
        """
        return self.anomaly_history[-limit:]


# Exemple d'intégration dans le gestionnaire de risque adaptatif:
# 
# Dans adaptive_risk_manager.py, remplacer la méthode _detect_extreme_market_conditions 