# strategies/hybrid_strategy.py
"""
Stratégie hybride combinant l'analyse technique classique avec des prédictions LSTM
"""
import os
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Union
from datetime import datetime, timedelta

from strategies.strategy_base import StrategyBase
from strategies.technical_bounce import TechnicalBounceStrategy
from ai.models.lstm_model import LSTMModel
from ai.models.feature_engineering import FeatureEngineering
from core.adaptive_risk_manager import AdaptiveRiskManager
from config.config import DATA_DIR
from utils.logger import setup_logger

logger = setup_logger("hybrid_strategy")

class HybridStrategy(StrategyBase):
    """
    Stratégie hybride qui combine:
    1. Détection classique de rebond technique
    2. Prédictions du modèle LSTM pour la direction, volatilité et momentum
    3. Gestion adaptative du risque
    """
    def __init__(self, data_fetcher, market_analyzer, scoring_engine,
                lstm_model: Optional[LSTMModel] = None,
                adaptive_risk_manager: Optional[AdaptiveRiskManager] = None):
        """
        Initialise la stratégie hybride
        
        Args:
            data_fetcher: Module de récupération des données
            market_analyzer: Analyseur d'état du marché
            scoring_engine: Moteur de scoring
            lstm_model: Modèle LSTM (chargé automatiquement si None)
            adaptive_risk_manager: Gestionnaire de risque adaptatif
        """
        super().__init__(data_fetcher, market_analyzer, scoring_engine)
        
        # Composants spécifiques à la stratégie hybride
        self.technical_strategy = TechnicalBounceStrategy(data_fetcher, market_analyzer, scoring_engine)
        self.feature_engineering = FeatureEngineering()
        self.adaptive_risk_manager = adaptive_risk_manager or AdaptiveRiskManager()
        
        # Chargement du modèle LSTM
        self.lstm_model = lstm_model
        if self.lstm_model is None:
            self._load_lstm_model()
        
        # Paramètres de combinaison des signaux
        self.lstm_weight = 0.6  # Poids des prédictions LSTM
        self.technical_weight = 0.4  # Poids des signaux techniques
        
        # Seuil minimum de score combiné pour trader
        self.min_score = 75
        
        # Cache des prédictions LSTM (pour éviter de recalculer)
        self.lstm_predictions_cache = {}
        self.cache_duration = 300  # 5 minutes de durée de cache
    
    def _load_lstm_model(self) -> None:
        """
        Charge le modèle LSTM depuis le disque
        """
        try:
            model_path = os.path.join(DATA_DIR, "models", "production", "lstm_final.h5")
            if not os.path.exists(model_path):
                logger.warning(f"Modèle LSTM non trouvé: {model_path}")
                return
            
            self.lstm_model = LSTMModel()
            self.lstm_model.load(model_path)
            logger.info(f"Modèle LSTM chargé: {model_path}")
        except Exception as e:
            logger.error(f"Erreur lors du chargement du modèle LSTM: {str(e)}")
    
    def find_trading_opportunity(self, symbol: str) -> Optional[Dict]:
        """
        Cherche une opportunité de trading en combinant les signaux techniques et les prédictions LSTM
        
        Args:
            symbol: Paire de trading
            
        Returns:
            Opportunité de trading ou None si aucune opportunité
        """
        # 1. Rechercher une opportunité selon la stratégie technique classique
        technical_opportunity = self.technical_strategy.find_trading_opportunity(symbol)
        
        # Si aucune opportunité technique, pas besoin d'aller plus loin
        if not technical_opportunity:
            return None
        
        # 2. Récupérer les données de marché
        market_data = self.data_fetcher.get_market_data(symbol)
        
        # 3. Obtenir les prédictions du modèle LSTM
        lstm_prediction = self._get_lstm_prediction(symbol, market_data)
        
        # Si aucune prédiction LSTM disponible, utiliser uniquement la stratégie technique
        if not lstm_prediction:
            logger.warning(f"Aucune prédiction LSTM disponible pour {symbol}")
            
            # Si le score technique est très élevé, on peut quand même trader
            if technical_opportunity["score"] >= 85:
                return technical_opportunity
            
            return None
        
        # 4. Combiner les signaux pour une décision finale
        combined_opportunity = self._combine_signals(
            symbol,
            technical_opportunity,
            lstm_prediction,
            market_data
        )
        
        return combined_opportunity
    
    def _get_lstm_prediction(self, symbol: str, market_data: Dict) -> Optional[Dict]:
        """
        Obtient les prédictions du modèle LSTM pour le symbole donné
        
        Args:
            symbol: Paire de trading
            market_data: Données de marché
            
        Returns:
            Prédictions LSTM ou None si indisponibles
        """
        # Vérifier si le modèle LSTM est disponible
        if self.lstm_model is None:
            return None
        
        # Vérifier si des prédictions récentes sont en cache
        cache_key = f"{symbol}_{datetime.now().strftime('%Y%m%d_%H%M')}"
        if cache_key in self.lstm_predictions_cache:
            cached_prediction = self.lstm_predictions_cache[cache_key]
            if (datetime.now() - cached_prediction["timestamp"]).total_seconds() < self.cache_duration:
                return cached_prediction["data"]
        
        try:
            # Récupérer les données OHLCV
            ohlcv_data = market_data["primary_timeframe"]["ohlcv"]
            
            # Créer les caractéristiques avancées
            featured_data = self.feature_engineering.create_features(
                ohlcv_data, 
                include_time_features=True,
                include_price_patterns=True
            )
            
            # Normaliser les caractéristiques
            normalized_data = self.feature_engineering.scale_features(
                featured_data,
                is_training=False,
                method='standard',
                feature_group='lstm'
            )
            
            # Créer une séquence pour la prédiction
            sequence_length = self.lstm_model.input_length
            
            # Vérifier si nous avons assez de données
            if len(normalized_data) < sequence_length:
                logger.warning(f"Données insuffisantes pour la prédiction LSTM ({len(normalized_data)} < {sequence_length})")
                return None
            
            # Obtenir la dernière séquence
            X = self.feature_engineering.prepare_lstm_data(
                normalized_data,
                sequence_length=sequence_length,
                is_training=False
            )
            
            # Faire la prédiction
            prediction = self.lstm_model.predict(normalized_data)
            
            # Stocker en cache
            self.lstm_predictions_cache[cache_key] = {
                "data": prediction,
                "timestamp": datetime.now()
            }
            
            # Nettoyer le cache (garder seulement les 10 prédictions les plus récentes)
            if len(self.lstm_predictions_cache) > 10:
                oldest_key = min(self.lstm_predictions_cache.keys(), 
                                key=lambda k: self.lstm_predictions_cache[k]["timestamp"])
                del self.lstm_predictions_cache[oldest_key]
            
            return prediction
        
        except Exception as e:
            logger.error(f"Erreur lors de la prédiction LSTM: {str(e)}")
            return None
    
    def _combine_signals(self, symbol: str, technical_opportunity: Dict, 
                       lstm_prediction: Dict, market_data: Dict) -> Optional[Dict]:
        """
        Combine les signaux techniques et les prédictions LSTM
        
        Args:
            symbol: Paire de trading
            technical_opportunity: Opportunité de la stratégie technique
            lstm_prediction: Prédictions du modèle LSTM
            market_data: Données de marché
            
        Returns:
            Opportunité combinée ou None si pas d'opportunité
        """
        # Extraire les informations pertinentes
        technical_score = technical_opportunity["score"]
        technical_side = technical_opportunity["side"]
        
        # Vérifier si les prédictions LSTM sont cohérentes avec la stratégie technique
        lstm_confidence = self._calculate_lstm_confidence(lstm_prediction, technical_side)
        
        # Calculer le score combiné
        combined_score = (technical_score * self.technical_weight + 
                          lstm_confidence["score"] * self.lstm_weight)
        
        # Si le score combiné est trop faible, pas d'opportunité
        if combined_score < self.min_score:
            return None
        
        # Fusionner les informations en une seule opportunité
        combined_opportunity = technical_opportunity.copy()
        combined_opportunity["score"] = combined_score
        combined_opportunity["lstm_prediction"] = lstm_prediction
        combined_opportunity["lstm_confidence"] = lstm_confidence
        
        # Ajuster les niveaux de stop-loss et take-profit en fonction des prédictions
        entry_price = technical_opportunity["entry_price"]
        
        # Utiliser le gestionnaire de risque adaptatif pour calculer les niveaux optimaux
        exit_levels = self.adaptive_risk_manager.calculate_optimal_exit_levels(
            entry_price,
            technical_side,
            technical_opportunity,
            lstm_prediction
        )
        
        combined_opportunity["stop_loss"] = exit_levels["stop_loss_price"]
        combined_opportunity["take_profit"] = exit_levels["take_profit_price"]
        combined_opportunity["stop_loss_percent"] = exit_levels["stop_loss_percent"]
        combined_opportunity["take_profit_percent"] = exit_levels["take_profit_percent"]
        
        # Ajouter des informations explicatives
        combined_opportunity["reasoning"] = self._generate_reasoning(
            technical_opportunity,
            lstm_prediction,
            lstm_confidence,
            combined_score
        )
        
        return combined_opportunity
    
    def _calculate_lstm_confidence(self, lstm_prediction: Dict, technical_side: str) -> Dict:
        """
        Calcule la confiance dans les prédictions LSTM et leur alignement avec la stratégie technique
        
        Args:
            lstm_prediction: Prédictions du modèle LSTM
            technical_side: Direction de la stratégie technique ('BUY'/'SELL')
            
        Returns:
            Dictionnaire avec le score de confiance et les détails
        """
        # Initialiser le score de confiance
        confidence_score = 50  # Score neutre par défaut
        
        # Vérifier l'alignement de la direction
        direction_alignment = 0
        direction_confidence = 0
        
        # Pour chaque horizon, vérifier la direction prédite
        for horizon, prediction in lstm_prediction.items():
            # Pondération selon l'horizon (plus de poids au court terme)
            if "horizon_12" in horizon:  # Court terme
                weight = 0.6
            elif "horizon_24" in horizon:  # Moyen terme
                weight = 0.3
            else:  # Long terme
                weight = 0.1
            
            direction_prob = prediction.get("direction_probability", 0.5)
            
            # Convertir la probabilité en score (0-100)
            # Pour BUY: direction_prob > 0.5 est favorable
            # Pour SELL: direction_prob < 0.5 est favorable
            if technical_side == "BUY":
                horizon_score = (direction_prob - 0.5) * 200 * weight  # -100 à +100, pondéré
            else:
                horizon_score = (0.5 - direction_prob) * 200 * weight  # -100 à +100, pondéré
            
            direction_alignment += horizon_score
            
            # Calculer la confiance dans la direction (indépendamment de l'alignement)
            confidence = abs(direction_prob - 0.5) * 2 * 100 * weight  # 0 à 100, pondéré
            direction_confidence += confidence
        
        # Ajuster le score en fonction de l'alignement de direction
        confidence_score += direction_alignment
        
        # Vérifier le momentum
        momentum_alignment = 0
        
        for horizon, prediction in lstm_prediction.items():
            # Pondération selon l'horizon
            if "horizon_12" in horizon:  # Court terme
                weight = 0.6
            elif "horizon_24" in horizon:  # Moyen terme
                weight = 0.3
            else:  # Long terme
                weight = 0.1
            
            momentum = prediction.get("predicted_momentum", 0)
            
            # Convertir le momentum en score (0-100)
            # Pour BUY: momentum > 0 est favorable
            # Pour SELL: momentum < 0 est favorable
            if technical_side == "BUY":
                horizon_score = momentum * 100 * weight  # -100 à +100, pondéré
            else:
                horizon_score = -momentum * 100 * weight  # -100 à +100, pondéré
            
            momentum_alignment += horizon_score
        
        # Ajuster le score en fonction de l'alignement de momentum
        confidence_score += momentum_alignment
        
        # Vérifier la volatilité
        volatility_factor = 0
        
        for horizon, prediction in lstm_prediction.items():
            # Pondération selon l'horizon
            if "horizon_12" in horizon:  # Court terme
                weight = 0.5
            elif "horizon_24" in horizon:  # Moyen terme
                weight = 0.3
            else:  # Long terme
                weight = 0.2
            
            volatility = prediction.get("predicted_volatility", 1.0)
            
            # Volatilité faible est généralement favorable pour les positions longues
            # Volatilité élevée peut être favorable pour les positions courtes
            if technical_side == "BUY":
                if volatility < 0.8:
                    volatility_score = 10 * weight
                elif volatility > 1.5:
                    volatility_score = -20 * weight
                else:
                    volatility_score = 0
            else:
                if volatility > 1.5:
                    volatility_score = 10 * weight
                elif volatility < 0.8:
                    volatility_score = -10 * weight
                else:
                    volatility_score = 0
            
            volatility_factor += volatility_score
        
        # Ajuster le score en fonction de la volatilité
        confidence_score += volatility_factor
        
        # Limiter le score final entre 0 et 100
        confidence_score = max(0, min(100, confidence_score))
        
        return {
            "score": confidence_score,
            "direction_alignment": direction_alignment,
            "direction_confidence": direction_confidence,
            "momentum_alignment": momentum_alignment,
            "volatility_factor": volatility_factor
        }
    
    def _generate_reasoning(self, technical_opportunity: Dict, 
                          lstm_prediction: Dict, 
                          lstm_confidence: Dict,
                          combined_score: float) -> str:
        """
        Génère une explication détaillée pour l'opportunité combinée
        
        Args:
            technical_opportunity: Opportunité de la stratégie technique
            lstm_prediction: Prédictions du modèle LSTM
            lstm_confidence: Confiance dans les prédictions LSTM
            combined_score: Score combiné
            
        Returns:
            Explication textuelle
        """
        # Extraire les signaux techniques
        technical_signals = technical_opportunity.get("signals", {}).get("signals", [])
        technical_score = technical_opportunity["score"]
        technical_side = technical_opportunity["side"]
        
        # Prendre les horizons court et moyen terme
        short_term = None
        mid_term = None
        
        for horizon, prediction in lstm_prediction.items():
            if "horizon_12" in horizon:
                short_term = prediction
            elif "horizon_24" in horizon:
                mid_term = prediction
        
        # Construire l'explication
        reasoning = f"Opportunité de trading {technical_side} détectée avec un score combiné de {combined_score:.1f}/100. "
        
        # Explication technique
        reasoning += f"Analyse technique ({technical_score:.1f} pts): "
        if technical_signals:
            reasoning += ", ".join(technical_signals[:3])
            if len(technical_signals) > 3:
                reasoning += f" et {len(technical_signals)-3} autres signaux"
        else:
            reasoning += "Signaux de rebond technique détectés"
        
        # Explication LSTM
        reasoning += f". Prédictions IA: "
        
        if short_term:
            direction_prob = short_term.get("direction_probability", 0.5) * 100
            momentum = short_term.get("predicted_momentum", 0)
            volatility = short_term.get("predicted_volatility", 1.0)
            
            # Direction
            if technical_side == "BUY":
                direction_text = f"{direction_prob:.1f}% de chance de hausse à court terme"
            else:
                direction_text = f"{(100-direction_prob):.1f}% de chance de baisse à court terme"
            
            # Momentum
            if abs(momentum) < 0.2:
                momentum_text = "momentum faible"
            elif abs(momentum) < 0.5:
                momentum_text = f"momentum {'positif' if momentum > 0 else 'négatif'} modéré"
            else:
                momentum_text = f"momentum {'positif' if momentum > 0 else 'négatif'} fort"
            
            # Volatilité
            if volatility < 0.8:
                volatility_text = "volatilité faible"
            elif volatility < 1.2:
                volatility_text = "volatilité normale"
            else:
                volatility_text = "volatilité élevée"
            
            reasoning += f"{direction_text}, {momentum_text}, {volatility_text}"
        
        # Ajouter des informations sur le mid-term si disponible
        if mid_term:
            direction_prob_mid = mid_term.get("direction_probability", 0.5) * 100
            
            if technical_side == "BUY":
                trend_coherence = "en cohérence" if direction_prob_mid > 50 else "en divergence"
            else:
                trend_coherence = "en cohérence" if direction_prob_mid < 50 else "en divergence"
            
            reasoning += f". Tendance à moyen terme {trend_coherence} ({direction_prob_mid:.1f}%)"
        
        # Ajouter des informations sur le risk/reward
        stop_loss = technical_opportunity.get("stop_loss", 0)
        take_profit = technical_opportunity.get("take_profit", 0)
        entry_price = technical_opportunity.get("entry_price", 0)
        
        if entry_price > 0 and stop_loss > 0 and take_profit > 0:
            if technical_side == "BUY":
                risk = (entry_price - stop_loss) / entry_price * 100
                reward = (take_profit - entry_price) / entry_price * 100
            else:
                risk = (stop_loss - entry_price) / entry_price * 100
                reward = (entry_price - take_profit) / entry_price * 100
            
            risk_reward_ratio = reward / risk if risk > 0 else 0
            
            reasoning += f". Ratio risque/récompense: {risk_reward_ratio:.2f} ({risk:.2f}% / {reward:.2f}%)"
        
        return reasoning
    
    def update_position_stops(self, symbol: str, position: Dict, current_price: float) -> Dict:
        """
        Met à jour les niveaux de stop-loss d'une position en utilisant les prédictions
        
        Args:
            symbol: Paire de trading
            position: Données de la position
            current_price: Prix actuel
            
        Returns:
            Nouvelles données de stop-loss
        """
        position_id = position.get("id", "unknown")
        
        # 1. Récupérer les données de marché
        market_data = self.data_fetcher.get_market_data(symbol)
        
        # 2. Obtenir les prédictions LSTM
        lstm_prediction = self._get_lstm_prediction(symbol, market_data)
        
        # 3. Mettre à jour les stops en fonction des prédictions
        stops_update = self.adaptive_risk_manager.calculate_position_dynamic_stops(
            position_id,
            current_price,
            position,
            lstm_prediction
        )
        
        return stops_update
    
    def should_close_early(self, symbol: str, position: Dict, current_price: float) -> Dict:
        """
        Détermine si une position doit être fermée prématurément
        
        Args:
            symbol: Paire de trading
            position: Données de la position
            current_price: Prix actuel
            
        Returns:
            Décision de fermeture anticipée
        """
        # Si le modèle LSTM n'est pas disponible, pas de fermeture anticipée
        if self.lstm_model is None:
            return {"should_close": False}
        
        position_id = position.get("id", "unknown")
        side = position.get("side", "BUY")
        entry_price = position.get("entry_price", current_price)
        
        # Calculer le profit actuel en pourcentage
        if side == "BUY":
            current_profit_pct = (current_price - entry_price) / entry_price * 100
        else:
            current_profit_pct = (entry_price - current_price) / entry_price * 100
        
        # 1. Récupérer les données de marché
        market_data = self.data_fetcher.get_market_data(symbol)
        
        # 2. Obtenir les prédictions LSTM
        lstm_prediction = self._get_lstm_prediction(symbol, market_data)
        
        # Si aucune prédiction disponible, ne pas fermer
        if not lstm_prediction:
            return {"should_close": False}
        
        # 3. Évaluer si la position doit être fermée
        should_close = False
        reason = ""
        
        # Extraire les prédictions de court terme
        short_term = None
        for horizon, prediction in lstm_prediction.items():
            if "horizon_12" in horizon:
                short_term = prediction
                break
        
        if short_term:
            direction_prob = short_term.get("direction_probability", 0.5)
            momentum = short_term.get("predicted_momentum", 0)
            
            # Pour les positions longues
            if side == "BUY":
                # Si forte probabilité de baisse et position en profit
                if direction_prob < 0.3 and momentum < -0.3 and current_profit_pct > 1:
                    should_close = True
                    reason = f"Forte probabilité de renversement baissier ({(1-direction_prob)*100:.1f}%)"
            
            # Pour les positions courtes
            else:
                # Si forte probabilité de hausse et position en profit
                if direction_prob > 0.7 and momentum > 0.3 and current_profit_pct > 1:
                    should_close = True
                    reason = f"Forte probabilité de renversement haussier ({direction_prob*100:.1f}%)"
        
        # 4. Vérifier les conditions de marché extrêmes
        extreme_conditions = self.adaptive_risk_manager._detect_extreme_market_conditions(market_data)
        if extreme_conditions["detected"] and current_profit_pct > 0:
            should_close = True
            reason = f"Conditions de marché extrêmes: {extreme_conditions['reason']}"
        
        return {
            "should_close": should_close,
            "reason": reason,
            "current_profit_pct": current_profit_pct
        }
    
    def get_market_prediction(self, symbol: str) -> Dict:
        """
        Fournit une prédiction de marché complète pour le tableau de bord
        
        Args:
            symbol: Paire de trading
            
        Returns:
            Prédiction complète du marché
        """
        market_data = self.data_fetcher.get_market_data(symbol)
        
        # Si le modèle LSTM n'est pas disponible, retourner une analyse technique standard
        if self.lstm_model is None:
            technical_analysis = self._get_technical_analysis(market_data)
            return {
                "symbol": symbol,
                "timestamp": datetime.now().isoformat(),
                "technical_analysis": technical_analysis,
                "lstm_available": False,
                "message": "Modèle LSTM non disponible, analyse technique uniquement"
            }
        
        # Obtenir les prédictions LSTM
        lstm_prediction = self._get_lstm_prediction(symbol, market_data)
        
        # Obtenir l'analyse technique
        technical_analysis = self._get_technical_analysis(market_data)
        
        # Combiner les analyses
        combined_analysis = self._combine_analysis(technical_analysis, lstm_prediction)
        
        return {
            "symbol": symbol,
            "timestamp": datetime.now().isoformat(),
            "technical_analysis": technical_analysis,
            "lstm_prediction": lstm_prediction,
            "combined_analysis": combined_analysis,
            "lstm_available": True
        }
    
    def _get_technical_analysis(self, market_data: Dict) -> Dict:
        """
        Effectue une analyse technique standard
        
        Args:
            market_data: Données de marché
            
        Returns:
            Résultat de l'analyse technique
        """
        # Extraire les indicateurs
        indicators = market_data.get("primary_timeframe", {}).get("indicators", {})
        
        # Analyse RSI
        rsi_analysis = "neutre"
        rsi_value = 50
        
        if "rsi" in indicators:
            rsi_value = float(indicators["rsi"].iloc[-1])
            if rsi_value < 30:
                rsi_analysis = "survente"
            elif rsi_value < 40:
                rsi_analysis = "baissier modéré"
            elif rsi_value > 70:
                rsi_analysis = "surachat"
            elif rsi_value > 60:
                rsi_analysis = "haussier modéré"
        
        # Analyse des bandes de Bollinger
        bb_analysis = "neutre"
        bb_position = 0.5
        
        if "bollinger" in indicators and "percent_b" in indicators["bollinger"]:
            bb_position = float(indicators["bollinger"]["percent_b"].iloc[-1])
            if bb_position < 0:
                bb_analysis = "sous-bande inférieure"
            elif bb_position < 0.2:
                bb_analysis = "proche de la bande inférieure"
            elif bb_position > 1:
                bb_analysis = "au-dessus de la bande supérieure"
            elif bb_position > 0.8:
                bb_analysis = "proche de la bande supérieure"
        
        # Analyse de la tendance (EMA)
        trend_analysis = "neutre"
        
        if "ema" in indicators:
            ema_short = indicators["ema"].get("ema_9", pd.Series()).iloc[-1] if "ema_9" in indicators["ema"] else None
            ema_medium = indicators["ema"].get("ema_21", pd.Series()).iloc[-1] if "ema_21" in indicators["ema"] else None
            ema_long = indicators["ema"].get("ema_50", pd.Series()).iloc[-1] if "ema_50" in indicators["ema"] else None
            
            if ema_short is not None and ema_medium is not None and ema_long is not None:
                if ema_short > ema_medium > ema_long:
                    trend_analysis = "haussier fort"
                elif ema_short > ema_medium:
                    trend_analysis = "haussier"
                elif ema_short < ema_medium < ema_long:
                    trend_analysis = "baissier fort"
                elif ema_short < ema_medium:
                    trend_analysis = "baissier"
        
        # Analyse ADX (force de tendance)
        adx_analysis = "tendance faible"
        adx_value = 0
        
        if "adx" in indicators:
            adx_data = indicators["adx"]
            adx_value = float(adx_data["adx"].iloc[-1])
            plus_di = float(adx_data["plus_di"].iloc[-1])
            minus_di = float(adx_data["minus_di"].iloc[-1])
            
            if adx_value > 25:
                if plus_di > minus_di:
                    adx_analysis = "tendance haussière forte"
                else:
                    adx_analysis = "tendance baissière forte"
            else:
                adx_analysis = "tendance faible"
        
        return {
            "trend": trend_analysis,
            "momentum": {
                "rsi": rsi_analysis,
                "rsi_value": rsi_value
            },
            "volatility": {
                "bollinger": bb_analysis,
                "bollinger_position": bb_position
            },
            "strength": {
                "adx": adx_analysis,
                "adx_value": adx_value
            },
            "summary": self._generate_technical_summary(trend_analysis, rsi_analysis, bb_analysis, adx_analysis)
        }
    
    def _generate_technical_summary(self, trend: str, rsi: str, bollinger: str, adx: str) -> Dict:
        """
        Génère un résumé de l'analyse technique
        
        Args:
            trend: Analyse de tendance
            rsi: Analyse RSI
            bollinger: Analyse Bollinger
            adx: Analyse ADX
            
        Returns:
            Résumé de l'analyse technique
        """
        # Calculer un score haussier/baissier
        bullish_score = 0
        bearish_score = 0
        
        # Évaluer la tendance
        if "haussier fort" in trend:
            bullish_score += 3
        elif "haussier" in trend:
            bullish_score += 2
        elif "baissier fort" in trend:
            bearish_score += 3
        elif "baissier" in trend:
            bearish_score += 2
        
        # Évaluer le RSI
        if "survente" in rsi:
            bullish_score += 2  # Potentiel rebond
        elif "surachat" in rsi:
            bearish_score += 2  # Potentiel repli
        elif "baissier" in rsi:
            bearish_score += 1
        elif "haussier" in rsi:
            bullish_score += 1
        
        # Évaluer les bandes de Bollinger
        if "sous-bande" in bollinger:
            bullish_score += 2  # Potentiel rebond
        elif "au-dessus" in bollinger:
            bearish_score += 2  # Potentiel repli
        elif "proche de la bande inférieure" in bollinger:
            bullish_score += 1
        elif "proche de la bande supérieure" in bollinger:
            bearish_score += 1
        
        # Évaluer l'ADX
        adx_multiplier = 1
        if "forte" in adx:
            adx_multiplier = 1.5
        
        # Conclusion
        total_bullish = bullish_score * adx_multiplier
        total_bearish = bearish_score * adx_multiplier
        
        bias = "neutre"
        if total_bullish > total_bearish * 1.5:
            bias = "fortement haussier"
        elif total_bullish > total_bearish:
            bias = "modérément haussier"
        elif total_bearish > total_bullish * 1.5:
            bias = "fortement baissier"
        elif total_bearish > total_bullish:
            bias = "modérément baissier"
        
        return {
            "bias": bias,
            "bullish_score": total_bullish,
            "bearish_score": total_bearish
        }
    
    def _combine_analysis(self, technical: Dict, lstm: Optional[Dict]) -> Dict:
        """
        Combine l'analyse technique et les prédictions LSTM
        
        Args:
            technical: Analyse technique
            lstm: Prédictions LSTM
            
        Returns:
            Analyse combinée
        """
        # Si pas de prédictions LSTM, retourner l'analyse technique
        if not lstm:
            return {
                "overall_bias": technical["summary"]["bias"],
                "confidence": "moyenne",
                "timeframes": {
                    "short_term": technical["summary"]["bias"],
                    "mid_term": "indéterminé",
                    "long_term": "indéterminé"
                },
                "explanation": "Basé uniquement sur l'analyse technique, LSTM non disponible"
            }
        
        # Extraire les prédictions par horizon
        short_term = None
        mid_term = None
        long_term = None
        
        for horizon, prediction in lstm.items():
            if "horizon_12" in horizon:
                short_term = prediction
            elif "horizon_24" in horizon:
                mid_term = prediction
            else:
                long_term = prediction
        
        # Déterminer le biais pour chaque horizon
        short_term_bias = "neutre"
        short_term_confidence = "faible"
        mid_term_bias = "neutre"
        long_term_bias = "neutre"
        
        if short_term:
            direction_prob = short_term.get("direction_probability", 0.5)
            momentum = short_term.get("predicted_momentum", 0)
            
            if direction_prob > 0.7:
                short_term_bias = "fortement haussier"
                short_term_confidence = "élevée"
            elif direction_prob > 0.6:
                short_term_bias = "modérément haussier"
                short_term_confidence = "moyenne"
            elif direction_prob < 0.3:
                short_term_bias = "fortement baissier"
                short_term_confidence = "élevée"
            elif direction_prob < 0.4:
                short_term_bias = "modérément baissier"
                short_term_confidence = "moyenne"
        
        if mid_term:
            direction_prob = mid_term.get("direction_probability", 0.5)
            
            if direction_prob > 0.65:
                mid_term_bias = "haussier"
            elif direction_prob < 0.35:
                mid_term_bias = "baissier"
        
        if long_term:
            direction_prob = long_term.get("direction_probability", 0.5)
            
            if direction_prob > 0.6:
                long_term_bias = "haussier"
            elif direction_prob < 0.4:
                long_term_bias = "baissier"
        
        # Combiner les analyses
        technical_bias = technical["summary"]["bias"]
        
        # Déterminer la cohérence entre technique et LSTM
        is_coherent = (
            ("haussier" in technical_bias and "haussier" in short_term_bias) or
            ("baissier" in technical_bias and "baissier" in short_term_bias)
        )
        
        overall_bias = "neutre"
        confidence = "moyenne"
        
        if is_coherent:
            # Si cohérent, renforcer le signal
            if "fortement" in technical_bias or "fortement" in short_term_bias:
                overall_bias = "fortement " + ("haussier" if "haussier" in short_term_bias else "baissier")
                confidence = "élevée"
            else:
                overall_bias = "modérément " + ("haussier" if "haussier" in short_term_bias else "baissier")
                confidence = "moyenne"
        else:
            # Si incohérent, favoriser légèrement les prédictions LSTM
            if short_term_confidence == "élevée":
                overall_bias = short_term_bias
                confidence = "moyenne"  # Réduite en raison de l'incohérence
            else:
                # Compromis
                overall_bias = "neutre avec tendance " + (
                    "haussière" if "haussier" in technical_bias or "haussier" in short_term_bias else "baissière"
                )
                confidence = "faible"
        
        # Générer une explication
        explanation = self._generate_combined_explanation(
            technical_bias, short_term_bias, mid_term_bias, is_coherent
        )
        
        return {
            "overall_bias": overall_bias,
            "confidence": confidence,
            "is_coherent": is_coherent,
            "timeframes": {
                "short_term": short_term_bias,
                "mid_term": mid_term_bias,
                "long_term": long_term_bias
            },
            "explanation": explanation
        }
    
    def _generate_combined_explanation(self, technical_bias: str, short_term_bias: str, 
                                     mid_term_bias: str, is_coherent: bool) -> str:
        """
        Génère une explication pour l'analyse combinée
        
        Args:
            technical_bias: Biais de l'analyse technique
            short_term_bias: Biais LSTM court terme
            mid_term_bias: Biais LSTM moyen terme
            is_coherent: Indique si les analyses sont cohérentes
            
        Returns:
            Explication textuelle
        """
        if is_coherent:
            explanation = f"L'analyse technique ({technical_bias}) est en accord avec les prédictions IA ({short_term_bias}), "
            
            if "haussier" in technical_bias:
                explanation += "suggérant un potentiel de hausse. "
            else:
                explanation += "indiquant une pression vendeuse. "
            
            if mid_term_bias != "neutre":
                explanation += f"Le moyen terme est également {mid_term_bias}. "
                
                if ("haussier" in short_term_bias and "haussier" in mid_term_bias) or \
                   ("baissier" in short_term_bias and "baissier" in mid_term_bias):
                    explanation += "La cohérence entre horizons renforce la fiabilité du signal."
                else:
                    explanation += "Attention à la divergence entre court et moyen terme."
        else:
            explanation = f"L'analyse technique ({technical_bias}) diverge des prédictions IA ({short_term_bias}). "
            
            explanation += "Cette divergence suggère une période d'incertitude. "
            
            if "haussier" in technical_bias:
                explanation += "Les indicateurs techniques montrent des signes haussiers, "
            else:
                explanation += "Les indicateurs techniques montrent des signes baissiers, "
                
            if "haussier" in short_term_bias:
                explanation += "tandis que l'IA prédit une tendance haussière à court terme. "
            else:
                explanation += "tandis que l'IA prédit une tendance baissière à court terme. "
            
            explanation += "Considérez une exposition réduite dans ce contexte contradictoire."
        
        return explanation