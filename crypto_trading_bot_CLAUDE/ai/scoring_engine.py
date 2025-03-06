# ai/scoring_engine.py
"""
Moteur de scoring pour évaluer les opportunités de trading
"""
import os
import json
import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Union
from datetime import datetime, timedelta

from config.config import DATA_DIR
from config.trading_params import LEARNING_RATE
from utils.logger import setup_logger

logger = setup_logger("scoring_engine")

class ScoringEngine:
    """
    Moteur de scoring qui évalue les opportunités de trading et s'améliore avec le temps
    """
    def __init__(self):
        self.weights = {}
        self.history = []
        self.weights_file = os.path.join(DATA_DIR, "ai_weights.json")
        self.history_file = os.path.join(DATA_DIR, "scoring_history.json")
        
        # Charger les poids et l'historique
        self._load_weights()
        self._load_history()
        
        # Initialiser les poids par défaut si nécessaire
        self._initialize_default_weights()
    
    def _load_weights(self) -> None:
        """
        Charge les poids depuis le fichier
        """
        if os.path.exists(self.weights_file):
            try:
                with open(self.weights_file, 'r') as f:
                    self.weights = json.load(f)
                logger.info("Poids chargés avec succès")
            except Exception as e:
                logger.error(f"Erreur lors du chargement des poids: {str(e)}")
                self.weights = {}
    
    def _load_history(self) -> None:
        """
        Charge l'historique de scoring depuis le fichier
        """
        if os.path.exists(self.history_file):
            try:
                with open(self.history_file, 'r') as f:
                    self.history = json.load(f)
                logger.info(f"Historique chargé: {len(self.history)} entrées")
            except Exception as e:
                logger.error(f"Erreur lors du chargement de l'historique: {str(e)}")
                self.history = []
    
    def _save_weights(self) -> None:
        """
        Sauvegarde les poids dans le fichier
        """
        try:
            with open(self.weights_file, 'w') as f:
                json.dump(self.weights, f, indent=2)
            logger.debug("Poids sauvegardés")
        except Exception as e:
            logger.error(f"Erreur lors de la sauvegarde des poids: {str(e)}")
    
    def _save_history(self) -> None:
        try:
            # Limiter la taille de l'historique avant de sauvegarder
            if len(self.history) > 1000:
                self.history = self.history[-1000:]
            
            with open(self.history_file, 'w') as f:
                json.dump(self.history, f, indent=2, default=str)
            logger.debug("Historique sauvegardé")
        except Exception as e:
            logger.error(f"Erreur lors de la sauvegarde de l'historique: {str(e)}")
    
    def _initialize_default_weights(self) -> None:
        """
        Initialise les poids par défaut si nécessaire
        """
        # Poids pour la stratégie de rebond technique
        if "technical_bounce" not in self.weights:
            self.weights["technical_bounce"] = {
                # Poids des signaux de rebond
            "rsi_oversold": 16,          
            "rsi_turning_up": 12,        
            "bollinger_below_lower": 16, 
            "bollinger_returning": 12,   
            "significant_lower_wick": 13, 
            "bullish_candle_after_bearish": 10, 
            "bullish_divergence": 21,
            "volume_spike": 12,           
            "adx_weak_trend": 6,          
            "no_strong_bearish_trend": 11, 
            "ema_alignment_not_bearish": 9, 
            "no_high_volatility": 6,     
            "stop_loss_percent": -12,     
            "risk_reward_ratio": 16   
            }
            
            self._save_weights()
            logger.info("Poids par défaut initialisés pour la stratégie de rebond technique")
    
    def calculate_score(self, data: Dict, strategy: str) -> Dict:
        """
        Calcule le score d'une opportunité de trading
        
        Args:
            data: Données pour le calcul du score
            strategy: Nom de la stratégie
            
        Returns:
            Dictionnaire avec le score et les détails
        """
        if strategy not in self.weights:
            logger.error(f"Stratégie non reconnue: {strategy}")
            return {"score": 0, "details": {}, "error": "Stratégie non reconnue"}
        
        # Calculer le score en fonction de la stratégie
        if strategy == "technical_bounce":
            return self._calculate_technical_bounce_score(data)
        
        return {"score": 0, "details": {}, "error": "Méthode de calcul non implémentée"}
    
    def _calculate_technical_bounce_score(self, data: Dict) -> Dict:
        """
        Calcule le score pour la stratégie de rebond technique
        
        Args:
            data: Données pour le calcul du score
            
        Returns:
            Dictionnaire avec le score et les détails
        """
    # Validation robuste des données d'entrée
        if not isinstance(data, dict):
            logger.error("Format de données invalide pour le scoring")
            return {"score": 0, "details": {}, "error": "Format de données invalide"}
        
        weights = self.weights.get("technical_bounce", {})
        if not weights:
            logger.error("Poids non initialisés pour la stratégie de rebond technique")
            return {"score": 0, "details": {}, "error": "Poids non initialisés"}
        
        score = 0
        details = {}
        
            # Extraire les données
        bounce_signals = data.get("bounce_signals", {})
        market_state = data.get("market_state", {})
        ohlcv = data.get("ohlcv", pd.DataFrame())
        indicators = data.get("indicators", {})
        
        # Valider les données
        if not bounce_signals or not market_state or ohlcv.empty:
            return {"score": 0, "details": {}, "error": "Données insuffisantes"}
        
        # 1. Évaluer les signaux de rebond
        signals = bounce_signals.get("signals", [])
        recent_market_performance = self._get_recent_market_performance(ohlcv)
        
        signal_weight_multiplier = 1.0
        if recent_market_performance < -5:  # Marché en forte baisse
            signal_weight_multiplier = 0.8  # Réduire l'importance des signaux haussiers
        elif recent_market_performance > 5:  # Marché en forte hausse
            signal_weight_multiplier = 1.2  # Augmenter l'importance des signaux haussiers
        
        # Application des poids pour chaque signal avec ajustement dynamique
        if "RSI en zone de survente" in signals:
            weight = weights["rsi_oversold"] * signal_weight_multiplier
            score += weight
            details["rsi_oversold"] = weight
        
        if "RSI remonte depuis la zone de survente" in signals:
            score += weights["rsi_turning_up"]
            details["rsi_turning_up"] = weights["rsi_turning_up"]
        
        if "Prix sous la bande inférieure de Bollinger" in signals:
            score += weights["bollinger_below_lower"]
            details["bollinger_below_lower"] = weights["bollinger_below_lower"]
        
        if "Prix remonte vers la bande inférieure" in signals:
            score += weights["bollinger_returning"]
            details["bollinger_returning"] = weights["bollinger_returning"]
        
        if "Mèche inférieure significative (rejet)" in signals:
            score += weights["significant_lower_wick"]
            details["significant_lower_wick"] = weights["significant_lower_wick"]
        
        if "Chandelier haussier après chandelier baissier" in signals:
            score += weights["bullish_candle_after_bearish"]
            details["bullish_candle_after_bearish"] = weights["bullish_candle_after_bearish"]
        
        if "Divergence haussière RSI détectée" in signals:
            score += weights["bullish_divergence"]
            details["bullish_divergence"] = weights["bullish_divergence"]
        
        if "Pic de volume haussier" in signals:
            score += weights["volume_spike"]
            details["volume_spike"] = weights["volume_spike"]
        
        # 2. Évaluer les conditions de marché
        market_details = market_state.get("details", {})
        
        # ADX faible (pas de tendance forte)
        if "adx" in market_details and market_details["adx"].get("value", 100) < 25:
            score += weights["adx_weak_trend"]
            details["adx_weak_trend"] = weights["adx_weak_trend"]
        
        # Pas de forte tendance baissière
        if "adx" in market_details and not (market_details["adx"].get("strong_trend", False) and market_details["adx"].get("bearish_trend", False)):
            score += weights["no_strong_bearish_trend"]
            details["no_strong_bearish_trend"] = weights["no_strong_bearish_trend"]
        
        # Alignement des EMA non baissier
        if "ema_alignment" in market_details and not market_details["ema_alignment"].get("bearish_alignment", False):
            score += weights["ema_alignment_not_bearish"]
            details["ema_alignment_not_bearish"] = weights["ema_alignment_not_bearish"]
        
        # Volatilité non excessive
        if "bollinger" in market_details and not market_details["bollinger"].get("high_volatility", False):
            score += weights["no_high_volatility"]
            details["no_high_volatility"] = weights["no_high_volatility"]

        contradictory_signals = self._detect_contradictory_signals(signals, indicators)
        if contradictory_signals:
            penalty = -15  # Pénalité significative
            score += penalty
            details["contradictory_signals_penalty"] = penalty
        
        # NOUVEAU: Bonus pour confirmation sur timeframes multiples
        multi_tf_confirmation = data.get("multi_timeframe_confirmation", 0)
        if multi_tf_confirmation > 0:
            bonus = multi_tf_confirmation * 5  # 5 points par timeframe confirmant
            score += bonus
            details["multi_timeframe_bonus"] = bonus
        # 3. Évaluer la qualité de l'opportunité
        entry_price = ohlcv["close"].iloc[-1]
        stop_loss_price = entry_price * 0.97  # -3% par défaut
        take_profit_price = entry_price * 1.06  # +6% par défaut
        
        # Si les prix sont fournis dans les données
        if "entry_price" in data and "stop_loss" in data and "take_profit" in data:
            entry_price = data["entry_price"]
            stop_loss_price = data["stop_loss"]
            take_profit_price = data["take_profit"]
        
        # Calculer le pourcentage de stop-loss
        stop_loss_percent = abs((entry_price - stop_loss_price) / entry_price * 100)
        
        # Pénaliser les stop-loss trop larges
        if stop_loss_percent > 5:
            penalty = weights["stop_loss_percent"] * (stop_loss_percent / 5)
            score += penalty  # Négatif
            details["stop_loss_penalty"] = penalty
        
        # Calculer le ratio risque/récompense
        risk = abs(entry_price - stop_loss_price)
        reward = abs(take_profit_price - entry_price)
        risk_reward_ratio = reward / risk if risk > 0 else 0
        
        # Bonus pour un bon ratio risque/récompense
        if risk_reward_ratio >= 1.5:
            bonus = weights["risk_reward_ratio"] * (risk_reward_ratio / 1.5)
            score += bonus
            details["risk_reward_bonus"] = bonus
        
        # 4. Normaliser le score (0-100)
        score = max(0, min(100, score))
        
        # 5. Enregistrer le résultat dans l'historique
        history_entry = {
            "timestamp": datetime.now().isoformat(),
            "strategy": "technical_bounce",
            "score": score,
            "details": details,
            "signals": signals,
            "trade_id": None,
            "market_context": {
                "trend": market_state.get("trend", "unknown"),
                "volatility": market_state.get("volatility", "medium")
            }
        }
        self.history.append(history_entry)
        self._save_history()
        if "RSI remonte depuis la zone de survente" in signals:
        # Vérifier la force du rebond RSI
            rsi_current = indicators.get("rsi", pd.Series()).iloc[-1]
            rsi_prev = indicators.get("rsi", pd.Series()).iloc[-2]
            
            rsi_momentum = rsi_current - rsi_prev
            if rsi_momentum > 5:  # Forte accélération du RSI
                score += weights["rsi_turning_up"] * 1.5
                details["strong_rsi_momentum"] = weights["rsi_turning_up"] * 1.5
        
        # Donnez plus de poids au volume lors des rebonds
        if "Pic de volume haussier" in signals:
            volume_ratio = bounce_signals.get("volume_ratio", 1.0)
            if volume_ratio > 3.0:  # Volume exceptionnellement élevé
                bonus = weights["volume_spike"] * (volume_ratio / 2)
                score += bonus
                details["high_volume_bonus"] = bonus

        # Ces instructions doivent être en dehors du bloc conditionnel
        self.history.append(history_entry)
        self._save_history()

        return {"score": int(score), "details": details}
    
    def _get_recent_market_performance(self, ohlcv: pd.DataFrame) -> float:
        """
        Calcule la performance récente du marché (pourcentage de changement sur les derniers jours)
        """
        if len(ohlcv) < 10:
            return 0
            
        # Calculer la performance sur les 5 derniers jours
        recent_close = ohlcv['close'].iloc[-1]
        past_close = ohlcv['close'].iloc[-10]
        
        return ((recent_close / past_close) - 1) * 100

    def _detect_contradictory_signals(self, signals: List[str], indicators: Dict) -> bool:
        """
        Détecte les signaux contradictoires qui pourraient indiquer un faux signal
        """
        # Exemple: RSI en zone de survente mais ADX fort avec tendance baissière
        if "RSI en zone de survente" in signals and indicators.get("adx", {}).get("strong_trend", False) and indicators.get("adx", {}).get("bearish_trend", False):
            return True
            
        # Exemple: Signal de rebond mais volume en baisse
        if "Chandelier haussier après chandelier baissier" in signals and not "Pic de volume haussier" in signals:
            return True
            
        return False 
    
    def update_trade_result(self, trade_id: str, trade_result: Dict) -> None:
        """
        Met à jour l'historique avec le résultat d'un trade et ajuste les poids
        
        Args:
            trade_id: ID du trade
            trade_result: Résultat du trade
        """
        # Rechercher l'entrée correspondante dans l'historique
        history_entry = None
        history_index = -1
        
        for i, entry in enumerate(reversed(self.history)):
            if entry.get("trade_id") == trade_id:
                history_entry = entry
                history_index = len(self.history) - 1 - i
                break
        
        if not history_entry:
            logger.warning(f"Entrée d'historique non trouvée pour le trade {trade_id}")
            return
        
        # Mettre à jour l'entrée avec le résultat
        self.history[history_index]["trade_result"] = trade_result
        self.history[history_index]["pnl_percent"] = trade_result.get("pnl_percent", 0)
        self.history[history_index]["pnl_absolute"] = trade_result.get("pnl_absolute", 0)
        
        # Ajuster les poids en fonction du résultat
        self._adjust_weights(history_index)
        
        # Sauvegarder l'historique et les poids
        self._save_history()
        self._save_weights()
    
    def _adjust_weights(self, history_index: int) -> None:
        """
        Ajuste les poids en fonction du résultat d'un trade avec mémoire adaptative
        
        Args:
            history_index: Index de l'entrée d'historique
        """
        if history_index < 0 or history_index >= len(self.history):
            logger.error(f"Index d'historique invalide: {history_index}")
            return
        
        history_entry = self.history[history_index]
        strategy = history_entry.get("strategy")
        pnl_percent = history_entry.get("pnl_percent", 0)
        
        if strategy not in self.weights:
            logger.error(f"Stratégie non reconnue: {strategy}")
            return
        
        # Ne pas ajuster les poids si le PnL est nul (trade non terminé)
        if pnl_percent == 0:
            return
        
        # Déterminer si le trade est un succès ou un échec
        is_success = pnl_percent > 0
        
        # Facteur d'ajustement basé sur la performance
        # Plus le gain ou la perte est importante, plus l'ajustement est grand
        adjustment_factor = abs(pnl_percent) / 5 * LEARNING_RATE
        adjustment_factor = min(adjustment_factor, 0.1)  # Limiter l'ajustement à 10% maximum
        
        # Facteur d'oubli pour les ajustements passés
        forget_factor = 0.85
        
        # Initialiser le dictionnaire des ajustements récents si nécessaire
        if not hasattr(self, 'recent_adjustments'):
            self.recent_adjustments = {}
        
        # Récupérer les détails du trade
        details = history_entry.get("details", {})
        weights = self.weights[strategy]
        
        # Ajuster les poids en fonction du succès ou de l'échec
        for factor, value in details.items():
            if factor in weights:
                # Récupérer l'ajustement précédent pour ce facteur (avec oubli)
                previous_adj = self.recent_adjustments.get(factor, 0) * forget_factor
                
                # Calculer l'ajustement actuel
                current_adj = adjustment_factor * (1 if is_success else -1)
                
                # Combiner les ajustements précédents et actuels
                total_adj = previous_adj + current_adj
                
                # Limiter l'ampleur totale de l'ajustement
                if abs(total_adj) > 0.15:
                    total_adj = 0.15 if total_adj > 0 else -0.15
                
                # Appliquer l'ajustement au poids
                weights[factor] = weights[factor] * (1 + total_adj)
                
                # Mémoriser cet ajustement pour les prochaines itérations
                self.recent_adjustments[factor] = total_adj
                
                logger.debug(f"Poids ajusté pour {factor}: {weights[factor]:.2f} (ajustement: {total_adj:.3f})")
        
        # Normaliser les poids pour éviter l'inflation ou la déflation
        total_weight = sum(abs(w) for w in weights.values())
        if total_weight > 0:
            scale_factor = 100 / total_weight
            for factor in weights:
                weights[factor] = weights[factor] * scale_factor
        
        logger.info(f"Poids ajustés pour la stratégie {strategy} (ajustement moyen: {adjustment_factor:.3f})")
        
        # Sauvegarder les poids mis à jour
        self._save_weights()

    def save_model_weights(self, filepath=None):
        """
        Saves model weights with robust error handling and backup creation
        
        Args:
            filepath: Path to save the weights, uses default if None
        """
        if filepath is None:
            filepath = os.path.join(self.models_dir, "scoring_weights.h5")
        
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        # Create backup of previous weights if they exist
        if os.path.exists(filepath):
            backup_path = filepath + f".backup_{int(time.time())}"
            try:
                shutil.copy2(filepath, backup_path)
                logger.info(f"Created backup of previous weights: {backup_path}")
            except Exception as e:
                logger.warning(f"Could not create backup of weights: {str(e)}")
        
        # Save weights with atomic write pattern to prevent corruption
        temp_filepath = filepath + ".tmp"
        try:
            self.model.save_weights(temp_filepath)
            
            # On Windows, we need to remove the target file first
            if os.path.exists(filepath):
                os.remove(filepath)
            
            os.rename(temp_filepath, filepath)
            
            logger.info(f"Successfully saved model weights to {filepath}")
            return True
        
        except Exception as e:
            logger.error(f"Error saving weights: {str(e)}")
            
            # If temporary file was created but not moved, clean it up
            if os.path.exists(temp_filepath):
                try:
                    os.remove(temp_filepath)
                except:
                    pass
            
            return False

    def save_history(self, history_data):
        """
        Saves training/prediction history with improved error handling
        
        Args:
            history_data: History data to save
        """
        history_path = os.path.join(self.data_dir, "scoring_history.json")
        
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(history_path), exist_ok=True)
        
        try:
            # Load existing history if available
            existing_history = []
            if os.path.exists(history_path):
                try:
                    with open(history_path, 'r') as f:
                        existing_history = json.load(f)
                except json.JSONDecodeError:
                    logger.warning(f"Could not decode existing history file. Starting new history.")
            
            # Append new data
            if not isinstance(existing_history, list):
                existing_history = []
            
            existing_history.append(history_data)
            
            # Keep only last 1000 entries to avoid file growth
            if len(existing_history) > 1000:
                existing_history = existing_history[-1000:]
            
            # Write with atomic pattern
            temp_path = history_path + ".tmp"
            with open(temp_path, 'w') as f:
                json.dump(existing_history, f, indent=2, default=str)
            
            # On Windows, we need to remove the target file first
            if os.path.exists(history_path):
                os.remove(history_path)
            
            os.rename(temp_path, history_path)
            
            logger.info(f"Successfully saved scoring history")
            return True
        
        except Exception as e:
            logger.error(f"Error saving history: {str(e)}")
            
            # Clean up temp file if it exists
            if os.path.exists(temp_path):
                try:
                    os.remove(temp_path)
                except:
                    pass
            
            return False
