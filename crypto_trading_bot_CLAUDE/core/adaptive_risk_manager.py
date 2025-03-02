# core/adaptive_risk_manager.py
"""
Gestionnaire de risque adaptatif avancé qui ajuste dynamiquement 
les paramètres de trading en fonction des conditions de marché,
des prédictions du modèle et de l'historique des trades
"""
import os
import json
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union
from datetime import datetime, timedelta
import math

from config.config import DATA_DIR
from config.trading_params import (
    RISK_PER_TRADE_PERCENT, 
    STOP_LOSS_PERCENT,
    TAKE_PROFIT_PERCENT,
    LEVERAGE
)
from ai.market_anomaly_detector import MarketAnomalyDetector
from utils.logger import setup_logger

logger = setup_logger("adaptive_risk_manager")

class AdaptiveRiskManager:
    """
    Gestionnaire de risque adaptatif qui ajuste dynamiquement les paramètres de trading
    
    Caractéristiques:
    - Ajuste la taille des positions en fonction de la confiance du modèle
    - Adapte les niveaux de stop-loss et take-profit selon la volatilité prédite
    - Devient plus conservateur après des séquences de pertes
    - Détecte les conditions de marché extrêmes et réduit l'exposition
    - Intègre un système anti-fragile qui apprend des pertes passées
    """
    def __init__(self, 
                initial_capital: float = 200,
                max_open_positions: int = 3,
                max_risk_per_day: float = 15.0,  # % max du capital risqué par jour
                recovery_factor: float = 0.5,     # Facteur de récupération après une perte
                enable_martingale: bool = False,   # Activer la stratégie de martingale (dangereux)
                enable_anti_martingale: bool = True,  # Stratégie anti-martingale (plus sûre)
                volatility_scaling: bool = True,   # Ajuster le risque selon la volatilité
                use_kelly_criterion: bool = True,  # Utiliser le critère de Kelly pour le sizing
                kelly_fraction: float = 0.5,      # Fraction du critère de Kelly (0.5 = demi-Kelly)
                risk_control_mode: str = "balanced"  # "conservative", "balanced", "aggressive"
                ):
        """
        Initialise le gestionnaire de risque adaptatif
        
        Args:
            initial_capital: Capital initial en USDT
            max_open_positions: Nombre max de positions ouvertes simultanément
            max_risk_per_day: Pourcentage max du capital risqué par jour
            recovery_factor: Facteur de réduction du risque après une perte
            enable_martingale: Active la stratégie de martingale (augmente le risque après une perte)
            enable_anti_martingale: Active la stratégie anti-martingale (augmente la taille après un gain)
            volatility_scaling: Ajuster la taille en fonction de la volatilité
            use_kelly_criterion: Utiliser le critère de Kelly pour le sizing optimal
            kelly_fraction: Fraction du critère de Kelly à utiliser (1.0 = Kelly complet)
            risk_control_mode: Mode de contrôle du risque (conservateur/équilibré/agressif)
        """
        # Paramètres de base
        self.initial_capital = initial_capital
        self.current_capital = initial_capital
        self.max_open_positions = max_open_positions
        self.max_risk_per_day = max_risk_per_day
        self.recovery_factor = recovery_factor
        self.enable_martingale = enable_martingale
        self.enable_anti_martingale = enable_anti_martingale
        self.volatility_scaling = volatility_scaling
        self.use_kelly_criterion = use_kelly_criterion
        self.kelly_fraction = kelly_fraction
        
        # Historique de trading
        self.trade_history = []
        self.daily_risk_used = 0.0
        self.last_risk_reset = datetime.now()
        self.consecutive_losses = 0
        self.consecutive_wins = 0
        
        # Profils de risque dynamiques
        self.risk_profiles = self._initialize_risk_profiles(risk_control_mode)
        self.current_risk_profile = "balanced"  # Commence en mode équilibré
        
        # Paramètres par défaut si aucun modèle LSTM n'est disponible
        self.default_params = {
            "risk_per_trade": RISK_PER_TRADE_PERCENT,
            "stop_loss": STOP_LOSS_PERCENT,
            "take_profit": TAKE_PROFIT_PERCENT,
            "leverage": LEVERAGE
        }
        
        # Indicateurs d'état du système
        self.market_state = "normal"  # normal, volatile, extreme
        self.risk_capacity = 1.0  # Facteur multiplicateur de risque (0.1 à 1.0)
        
        # Détecteur d'anomalies de marché
        self.anomaly_detector = MarketAnomalyDetector(
            model_dir=os.path.join(DATA_DIR, "models", "anomaly")
        )
        
        # Initialiser le journal pour la traçabilité des décisions
        self.risk_log = []
        
        # Charger l'historique précédent si disponible
        self._load_history()
    
    def _initialize_risk_profiles(self, mode: str) -> Dict:
        """
        Initialise les profils de risque pour différentes conditions de marché
        
        Args:
            mode: Mode de contrôle du risque (conservative/balanced/aggressive)
            
        Returns:
            Dictionnaire des profils de risque
        """
        # Base des profils de risque
        profiles = {
            # Profil par défaut/équilibré
            "balanced": {
                "risk_per_trade_percent": RISK_PER_TRADE_PERCENT,
                "stop_loss_percent": STOP_LOSS_PERCENT,
                "take_profit_percent": TAKE_PROFIT_PERCENT,
                "leverage": LEVERAGE,
                "trailing_stop_activation": 2.0,  # % de profit pour activer le trailing stop
                "trailing_stop_step": 0.5,        # % de distance pour le trailing stop
                "risk_scaling_factor": 1.0        # Facteur d'échelle pour le risque
            },
            
            # Profil conservateur pour les conditions volatiles
            "conservative": {
                "risk_per_trade_percent": RISK_PER_TRADE_PERCENT * 0.5,
                "stop_loss_percent": STOP_LOSS_PERCENT * 1.25,  # Stop loss plus large
                "take_profit_percent": TAKE_PROFIT_PERCENT * 1.25, # TP plus large
                "leverage": max(1, LEVERAGE - 1),    # Levier réduit
                "trailing_stop_activation": 1.5,
                "trailing_stop_step": 0.3,
                "risk_scaling_factor": 0.5
            },
            
            # Profil ultra-conservateur pour les conditions extrêmes
            "defensive": {
                "risk_per_trade_percent": RISK_PER_TRADE_PERCENT * 0.25,
                "stop_loss_percent": STOP_LOSS_PERCENT * 1.5,
                "take_profit_percent": TAKE_PROFIT_PERCENT * 1.5,
                "leverage": 1,  # Pas de levier en mode défensif
                "trailing_stop_activation": 1.0,
                "trailing_stop_step": 0.2,
                "risk_scaling_factor": 0.25
            },
            
            # Profil agressif pour les conditions favorables
            "aggressive": {
                "risk_per_trade_percent": min(10.0, RISK_PER_TRADE_PERCENT * 1.25),
                "stop_loss_percent": STOP_LOSS_PERCENT * 0.8,
                "take_profit_percent": TAKE_PROFIT_PERCENT * 0.8,
                "leverage": min(5, LEVERAGE + 1),
                "trailing_stop_activation": 3.0,
                "trailing_stop_step": 0.8,
                "risk_scaling_factor": 1.25
            },
            
            # Profil très agressif pour les conditions très favorables
            "very_aggressive": {
                "risk_per_trade_percent": min(12.5, RISK_PER_TRADE_PERCENT * 1.5),
                "stop_loss_percent": STOP_LOSS_PERCENT * 0.7,
                "take_profit_percent": TAKE_PROFIT_PERCENT * 0.7,
                "leverage": min(5, LEVERAGE + 2),
                "trailing_stop_activation": 4.0,
                "trailing_stop_step": 1.0,
                "risk_scaling_factor": 1.5
            }
        }
        
        # Ajuster les profils en fonction du mode sélectionné
        if mode == "conservative":
            # Rendre tous les profils plus conservateurs
            for profile in profiles.values():
                profile["risk_per_trade_percent"] *= 0.8
                profile["leverage"] = max(1, profile["leverage"] - 1)
                profile["risk_scaling_factor"] *= 0.8
        
        elif mode == "aggressive":
            # Rendre tous les profils plus agressifs
            for profile in profiles.values():
                profile["risk_per_trade_percent"] *= 1.2
                profile["leverage"] = min(5, profile["leverage"] + 1)
                profile["risk_scaling_factor"] *= 1.2
        
        return profiles
    
    def update_account_balance(self, account_info: Dict) -> None:
        """
        Met à jour le capital disponible
        
        Args:
            account_info: Informations sur le compte
        """
        # Mettre à jour le capital
        if "totalWalletBalance" in account_info:
            # Pour Binance Futures
            self.current_capital = float(account_info["totalWalletBalance"])
        elif "totalBalance" in account_info:
            # Pour Binance Spot
            self.current_capital = float(account_info["totalBalance"])
        else:
            logger.warning("Format d'information de compte non reconnu")
        
        logger.info(f"Capital mis à jour: {self.current_capital} USDT")
        
        # Réinitialiser le risque quotidien si nécessaire
        current_time = datetime.now()
        if (current_time - self.last_risk_reset).days >= 1:
            self.daily_risk_used = 0.0
            self.last_risk_reset = current_time
            logger.info("Limite de risque quotidien réinitialisée")
    
    def can_open_new_position(self, position_tracker) -> Dict:
        """
        Vérifie si une nouvelle position peut être ouverte selon les règles de gestion des risques
        
        Args:
            position_tracker: Objet qui suit les positions ouvertes
            
        Returns:
            Dictionnaire avec décision et raison
        """
        # 1. Vérifier le nombre de positions ouvertes
        open_positions = position_tracker.get_all_open_positions()
        total_open_positions = sum(len(positions) for positions in open_positions.values())
        
        if total_open_positions >= self.max_open_positions:
            return {
                "can_open": False,
                "reason": f"Nombre maximum de positions atteint ({self.max_open_positions})"
            }
        
        # 2. Vérifier le risque quotidien utilisé
        max_risk_amount = self.current_capital * (self.max_risk_per_day / 100)
        
        if self.daily_risk_used >= max_risk_amount:
            return {
                "can_open": False,
                "reason": f"Limite de risque quotidien atteinte ({self.max_risk_per_day}% du capital)"
            }
        
        # 3. Vérifier l'état du marché
        if self.market_state == "extreme":
            # En conditions extrêmes, être plus restrictif
            if self.current_risk_profile != "defensive":
                self.set_risk_profile("defensive")
                
            if total_open_positions > 0:
                return {
                    "can_open": False,
                    "reason": "Conditions de marché extrêmes - aucune nouvelle position"
                }
        
        # 4. Ajuster en fonction de l'historique récent
        if self.consecutive_losses >= 3:
            # Après 3 pertes consécutives, être plus conservateur
            if self.current_risk_profile not in ["conservative", "defensive"]:
                self.set_risk_profile("conservative")
                logger.info("Passage en mode conservateur après 3 pertes consécutives")
        
        elif self.consecutive_wins >= 3:
            # Après 3 gains consécutifs, être légèrement plus agressif
            if self.current_risk_profile == "balanced":
                self.set_risk_profile("aggressive")
                logger.info("Passage en mode agressif après 3 gains consécutifs")
        
        # 5. Vérifier le capital minimum
        min_capital_required = 50  # Montant minimum pour trader raisonnablement
        
        if self.current_capital < min_capital_required:
            return {
                "can_open": False,
                "reason": f"Capital insuffisant ({self.current_capital} < {min_capital_required} USDT)"
            }
        
        return {
            "can_open": True,
            "risk_profile": self.current_risk_profile,
            "available_risk": max_risk_amount - self.daily_risk_used
        }
    
    def calculate_position_size(self, symbol: str, opportunity: Dict, 
                              lstm_prediction: Optional[Dict] = None) -> float:
        """
        Calcule la taille optimale de position en fonction de multiples facteurs
        
        Args:
            symbol: Paire de trading
            opportunity: Opportunité de trading détectée
            lstm_prediction: Prédictions du modèle LSTM (optionnel)
            
        Returns:
            Taille de position en USDT
        """
        # Obtenir les paramètres du profil de risque actuel
        profile = self.risk_profiles[self.current_risk_profile]
        
        # Paramètres de base
        base_risk_percent = profile["risk_per_trade_percent"]
        risk_factor = profile["risk_scaling_factor"]
        stop_loss_percent = profile["stop_loss_percent"]
        leverage = profile["leverage"]
        
        # 1. Ajuster le risque en fonction du score de l'opportunité
        score = opportunity.get("score", 70)
        score_factor = self._calculate_score_factor(score)
        
        # 2. Intégrer les prédictions LSTM si disponibles
        model_confidence = 0.5  # Valeur par défaut (neutre)
        volatility_factor = 1.0
        
        if lstm_prediction:
            # Trouver l'horizon pertinent (court terme pour le sizing)
            short_term = None
            for horizon_name, prediction in lstm_prediction.items():
                if horizon_name in ["3h", "4h", "short_term", "horizon_12"]:
                    short_term = prediction
                    break
            
            if short_term:
                # Confiance dans la direction prédite
                direction_prob = short_term.get("direction_probability", 50) / 100
                
                # Ajuster la confiance en fonction de l'écart par rapport à 0.5
                model_confidence = abs(direction_prob - 0.5) * 2
                
                # Si la direction prédite est opposée à celle de l'opportunité, réduire la taille
                predicted_direction = direction_prob > 0.5
                opportunity_direction = opportunity.get("side", "BUY") == "BUY"
                
                if predicted_direction != opportunity_direction:
                    model_confidence = 0.2  # Faible confiance en cas de contradiction
                
                # Ajuster en fonction de la volatilité prédite
                predicted_volatility = short_term.get("predicted_volatility", 0.03)
                volatility_factor = self._volatility_adjustment(predicted_volatility)
        
        # 3. Calcul de la taille de base (basée sur le risque)
        risk_amount = self.current_capital * (base_risk_percent / 100) * risk_factor
        
        # 4. Appliquer le facteur de Kelly si activé
        if self.use_kelly_criterion and len(self.trade_history) >= 10:
            kelly_size = self._calculate_kelly_criterion()
            # Appliquer la fraction de Kelly
            kelly_adjusted_risk = risk_amount * kelly_size * self.kelly_fraction
            risk_amount = min(risk_amount, kelly_adjusted_risk)
        
        # 5. Ajuster avec les facteurs de confiance
        confidence_factor = (score_factor + model_confidence) / 2
        adjusted_risk = risk_amount * confidence_factor
        
        # 6. Appliquer le facteur de volatilité si activé
        if self.volatility_scaling:
            adjusted_risk *= volatility_factor
        
        # 7. Appliquer les stratégies de martingale/anti-martingale si activées
        if self.consecutive_losses > 0 and self.enable_martingale:
            # Augmenter légèrement la taille après une perte (risqué!)
            martingale_factor = 1.0 + (0.1 * min(self.consecutive_losses, 3))
            adjusted_risk *= martingale_factor
        
        elif self.consecutive_wins > 0 and self.enable_anti_martingale:
            # Augmenter la taille après un gain
            anti_martingale_factor = 1.0 + (0.15 * min(self.consecutive_wins, 3))
            adjusted_risk *= anti_martingale_factor
        
        # 8. Ajuster en fonction de la capacity de risque globale
        adjusted_risk *= self.risk_capacity
        
        # 9. Limites de sécurité pour éviter les overrides manuels
        # Plafond à 15% du capital quelle que soit la configuration
        max_risk_allowed = self.current_capital * 0.15
        adjusted_risk = min(adjusted_risk, max_risk_allowed)
        
        # Calculer la taille finale
        # Position size = (Capital * Risk%) / (StopLoss% * Leverage)
        position_size = adjusted_risk / (stop_loss_percent / 100) * leverage
        
        # Limiter la taille aux fonds disponibles
        position_size = min(position_size, self.current_capital * 0.95)
        
        # Mettre à jour le risque quotidien utilisé
        self.daily_risk_used += adjusted_risk
        
        # Journaliser la décision
        self._log_position_sizing(
            symbol=symbol,
            base_risk=risk_amount,
            final_size=position_size,
            factors={
                "score_factor": score_factor,
                "model_confidence": model_confidence,
                "volatility_factor": volatility_factor,
                "risk_capacity": self.risk_capacity,
                "kelly_criterion": self.use_kelly_criterion,
                "profile": self.current_risk_profile
            }
        )
        
        return position_size
    
    def update_stop_loss(self, symbol: str, original_stop: float, 
                       current_price: float, lstm_prediction: Dict) -> Dict:
        """
        Calcule un niveau de stop-loss adaptatif basé sur la volatilité prédite
        
        Args:
            symbol: Paire de trading
            original_stop: Niveau de stop-loss original
            current_price: Prix actuel
            lstm_prediction: Prédictions du modèle LSTM
            
        Returns:
            Nouveau niveau de stop-loss et raisonnement
        """
        # Si pas de prédiction LSTM disponible, conserver le stop original
        if not lstm_prediction:
            return {
                "stop_level": original_stop,
                "updated": False,
                "reason": "Aucune prédiction LSTM disponible"
            }
        
        # Trouver l'horizon pertinent (court terme)
        short_term = None
        for horizon_name, prediction in lstm_prediction.items():
            if horizon_name in ["3h", "4h", "short_term", "horizon_12"]:
                short_term = prediction
                break
        
        if not short_term:
            return {
                "stop_level": original_stop,
                "updated": False,
                "reason": "Aucune prédiction court terme disponible"
            }
        
        # Récupérer la volatilité prédite
        predicted_volatility = short_term.get("predicted_volatility", 0.03)
        
        # Calculer la différence en pourcentage entre le prix actuel et le stop original
        original_stop_percent = abs((current_price - original_stop) / current_price * 100)
        
        # Ajuster en fonction de la volatilité prédite
        # Plus la volatilité est élevée, plus le stop doit être large
        volatility_percent = predicted_volatility * 100
        
        # Facteur d'ajustement: si la volatilité prédite est élevée, augmenter le stop
        # si elle est faible, réduire le stop
        adjustment_factor = volatility_percent / 3.0  # 3% est considéré comme une volatilité standard
        
        # Calculer le nouveau stop en pourcentage
        new_stop_percent = original_stop_percent * adjustment_factor
        
        # Limites de sécurité
        min_stop_percent = 1.0  # Minimum 1%
        max_stop_percent = 10.0  # Maximum 10%
        
        new_stop_percent = max(min_stop_percent, min(new_stop_percent, max_stop_percent))
        
        # Calculer le nouveau niveau de stop-loss
        side = "BUY" if current_price > original_stop else "SELL"
        
        if side == "BUY":
            # Pour les positions longues, le stop est en dessous du prix
            new_stop_level = current_price * (1 - new_stop_percent / 100)
        else:
            # Pour les positions courtes, le stop est au-dessus du prix
            new_stop_level = current_price * (1 + new_stop_percent / 100)
        
        # Ne pas déplacer le stop dans la mauvaise direction
        if side == "BUY" and new_stop_level < original_stop:
            new_stop_level = original_stop
        elif side == "SELL" and new_stop_level > original_stop:
            new_stop_level = original_stop
        
        return {
            "stop_level": new_stop_level,
            "updated": new_stop_level != original_stop,
            "reason": f"Ajustement basé sur volatilité prédite de {volatility_percent:.2f}%",
            "volatility_percent": volatility_percent,
            "adjustment_factor": adjustment_factor
        }
    
    def update_take_profit(self, symbol: str, original_tp: float, 
                         current_price: float, lstm_prediction: Dict) -> Dict:
        """
        Calcule un niveau de take-profit adaptatif basé sur le momentum et la volatilité prédits
        
        Args:
            symbol: Paire de trading
            original_tp: Niveau de take-profit original
            current_price: Prix actuel
            lstm_prediction: Prédictions du modèle LSTM
            
        Returns:
            Nouveau niveau de take-profit et raisonnement
        """
        # Si pas de prédiction LSTM disponible, conserver le TP original
        if not lstm_prediction:
            return {
                "tp_level": original_tp,
                "updated": False,
                "reason": "Aucune prédiction LSTM disponible"
            }
        
        # Récupérer les prédictions à court et moyen terme
        short_term = None
        medium_term = None
        
        for horizon_name, prediction in lstm_prediction.items():
            if horizon_name in ["3h", "4h", "short_term", "horizon_12"]:
                short_term = prediction
            elif horizon_name in ["12h", "24h", "medium_term", "horizon_48"]:
                medium_term = prediction
        
        if not short_term or not medium_term:
            return {
                "tp_level": original_tp,
                "updated": False,
                "reason": "Prédictions insuffisantes"
            }
        
        # Récupérer les indicateurs pertinents
        short_momentum = short_term.get("predicted_momentum", 0.0)
        medium_momentum = medium_term.get("predicted_momentum", 0.0)
        
        # Moyenne pondérée du momentum (plus de poids au court terme)
        momentum_score = (short_momentum * 0.7 + medium_momentum * 0.3)
        
        # Volatilité prédite
        volatility = short_term.get("predicted_volatility", 0.03) * 100  # en pourcentage
        
        # Calculer la différence en pourcentage entre le prix actuel et le TP original
        original_tp_percent = abs((original_tp - current_price) / current_price * 100)
        
        # Ajuster en fonction du momentum et de la volatilité
        # Plus le momentum est fort, plus le TP peut être agressif
        momentum_factor = 1.0 + (abs(momentum_score) * 0.5)
        
        # Ajuster également en fonction de la volatilité
        volatility_factor = max(0.8, min(1.5, volatility / 3.0))
        
        # Calculer le nouveau TP en pourcentage
        new_tp_percent = original_tp_percent * momentum_factor * volatility_factor
        
        # Limites de sécurité
        min_tp_percent = 2.0  # Minimum 2%
        max_tp_percent = 15.0  # Maximum 15%
        
        new_tp_percent = max(min_tp_percent, min(new_tp_percent, max_tp_percent))
        
        # Calculer le nouveau niveau de take-profit
        side = "BUY" if original_tp > current_price else "SELL"
        
        if side == "BUY":
            # Pour les positions longues, le TP est au-dessus du prix
            new_tp_level = current_price * (1 + new_tp_percent / 100)
        else:
            # Pour les positions courtes, le TP est en dessous du prix
            new_tp_level = current_price * (1 - new_tp_percent / 100)
        
        # Ne pas déplacer le TP dans la mauvaise direction
        if side == "BUY" and new_tp_level < original_tp:
            new_tp_level = original_tp
        elif side == "SELL" and new_tp_level > original_tp:
            new_tp_level = original_tp
        
        return {
            "tp_level": new_tp_level,
            "updated": new_tp_level != original_tp,
            "reason": f"Ajustement basé sur momentum ({momentum_score:.2f}) et volatilité ({volatility:.2f}%)",
            "momentum_score": momentum_score,
            "volatility_percent": volatility,
            "momentum_factor": momentum_factor,
            "volatility_factor": volatility_factor
        }
    
    def detect_extreme_market_conditions(self, data_fetcher, symbol: str) -> Dict:
        """
        Détecte des conditions de marché extrêmes qui nécessitent une réduction de l'exposition
        
        Args:
            data_fetcher: Instance du gestionnaire de données de marché
            symbol: Paire de trading
            
        Returns:
            Résultat de la détection
        """
        try:
            # Récupérer les données de marché
            market_data = data_fetcher.get_market_data(symbol)
            
            # Utiliser le détecteur d'anomalies
            anomaly_result = self.anomaly_detector.detect_anomalies(
                market_data["primary_timeframe"]["ohlcv"],
                current_price=market_data["current_price"],
                return_details=True
            )
            
            if anomaly_result["detected"]:
                # Condition de marché extrême détectée
                self.market_state = "extreme"
                
                # Ajuster le profil de risque
                self.set_risk_profile("defensive")
                
                # Réduire la capacité de risque
                self.risk_capacity = 0.2  # Réduction drastique
                
                return {
                    "extreme_condition": True,
                    "reason": anomaly_result["reason"],
                    "action_taken": "Passage en mode défensif",
                    "risk_capacity": self.risk_capacity,
                    "anomaly_details": anomaly_result["details"]
                }
            else:
                # Vérifier si c'est juste volatile mais pas extrême
                volatility_detected = False
                
                # Vérifier les indicateurs de volatilité
                atr_percent = market_data["primary_timeframe"]["indicators"]["atr"][-1] / market_data["current_price"] * 100
                
                if atr_percent > 5.0:  # 5% est considéré comme très volatile
                    volatility_detected = True
                    self.market_state = "volatile"
                    self.set_risk_profile("conservative")
                    self.risk_capacity = 0.5
                else:
                    # Conditions normales
                    self.market_state = "normal"
                    
                    # Si on était en mode défensif, revenir en mode équilibré
                    if self.current_risk_profile == "defensive":
                        self.set_risk_profile("conservative")
                    
                    # Restaurer progressivement la capacité de risque
                    self.risk_capacity = min(1.0, self.risk_capacity + 0.1)
                
                return {
                    "extreme_condition": False,
                    "volatile_condition": volatility_detected,
                    "market_state": self.market_state,
                    "risk_capacity": self.risk_capacity
                }
        
        except Exception as e:
            logger.error(f"Erreur lors de la détection des conditions extrêmes: {str(e)}")
            return {
                "extreme_condition": False,
                "error": str(e)
            }
    
    def should_close_early(self, symbol: str, position: Dict, current_price: float,
                         lstm_prediction: Optional[Dict] = None) -> Dict:
        """
        Détermine si une position doit être fermée de manière anticipée
        en fonction des prédictions du modèle ou des conditions de marché
        
        Args:
            symbol: Paire de trading
            position: Données de la position ouverte
            current_price: Prix actuel
            lstm_prediction: Prédictions du modèle LSTM
            
        Returns:
            Décision de fermeture anticipée avec raison
        """
        # Récupérer les détails de la position
        side = position.get("side", "BUY")
        entry_price = position.get("entry_price", current_price)
        position_age = datetime.now() - datetime.fromisoformat(position.get("entry_time", datetime.now().isoformat()))
        
        # Calculer le profit actuel
        if side == "BUY":
            profit_percent = (current_price - entry_price) / entry_price * 100 * position.get("leverage", 1)
        else:
            profit_percent = (entry_price - current_price) / entry_price * 100 * position.get("leverage", 1)
        
        # 1. Vérifier les conditions de marché extrêmes
        if self.market_state == "extreme":
            # Fermer avec profit ou petite perte
            if profit_percent > 0 or profit_percent > -1.5:
                return {
                    "should_close": True,
                    "reason": f"Conditions de marché extrêmes ({self.market_state})"
                }
        
        # 2. Utiliser les prédictions LSTM si disponibles
        if lstm_prediction:
            reversal_alert = lstm_prediction.get("reversal_alert", {})
            reversal_probability = reversal_alert.get("probability", 0.0)
            
            # Fermeture en cas d'alerte de retournement forte
            if reversal_probability > 0.8 and profit_percent > 0:
                return {
                    "should_close": True,
                    "reason": f"Alerte de retournement imminente (probabilité: {reversal_probability:.2f})"
                }
            
            # Vérifier si la prédiction est maintenant contre la position
            for horizon_name, prediction in lstm_prediction.items():
                if horizon_name in ["3h", "4h", "short_term", "horizon_12"]:
                    direction_prob = prediction.get("direction_probability", 50) / 100
                    direction_contradicts = (side == "BUY" and direction_prob < 0.3) or (side == "SELL" and direction_prob > 0.7)
                    
                    if direction_contradicts and profit_percent > 1.0:
                        return {
                            "should_close": True,
                            "reason": f"Prédiction de retournement à court terme (direction: {direction_prob:.2f})"
                        }
        
        # 3. Lock in profits pour les positions très profitables
        if profit_percent > 10.0:
            return {
                "should_close": True,
                "reason": f"Sécurisation du profit exceptionnel ({profit_percent:.2f}%)"
            }
        
        # 4. Fermeture des positions en stagnation prolongée
        stagnation_hours = 24  # Considérer la fermeture après 24h sans progrès
        if position_age.total_seconds() / 3600 > stagnation_hours and -1.0 < profit_percent < 2.0:
            return {
                "should_close": True,
                "reason": f"Position en stagnation après {position_age.total_seconds()/3600:.1f}h"
            }
        
        # Aucune raison de fermer maintenant
        return {
            "should_close": False,
            "current_profit": profit_percent,
            "position_age_hours": position_age.total_seconds() / 3600
        }
    
    def update_after_trade_closed(self, trade_result: Dict) -> None:
        """
        Met à jour l'état interne après qu'un trade a été fermé
        
        Args:
            trade_result: Résultat du trade fermé
        """
        # Extraire les résultats du trade
        pnl_percent = trade_result.get("pnl_percent", 0.0)
        pnl_absolute = trade_result.get("pnl_absolute", 0.0)
        
        # Mettre à jour le capital
        self.current_capital += pnl_absolute
        
        # Mettre à jour l'historique des trades
        self.trade_history.append({
            "timestamp": datetime.now().isoformat(),
            "pnl_percent": pnl_percent,
            "pnl_absolute": pnl_absolute,
            "risk_profile": self.current_risk_profile
        })
        
        # Limiter la taille de l'historique
        if len(self.trade_history) > 100:
            self.trade_history = self.trade_history[-100:]
        
        # Mettre à jour les compteurs de victoires/défaites consécutives
        if pnl_percent > 0:
            self.consecutive_wins += 1
            self.consecutive_losses = 0
        else:
            self.consecutive_losses += 1
            self.consecutive_wins = 0
        
        # Ajuster le profil de risque si nécessaire
        self._adjust_risk_profile_based_on_performance()
        
        # Sauvegarder l'historique
        self._save_history()
    
    def set_risk_profile(self, profile_name: str) -> bool:
        """
        Change explicitement le profil de risque
        
        Args:
            profile_name: Nom du profil de risque
            
        Returns:
            Succès du changement
        """
        if profile_name in self.risk_profiles:
            self.current_risk_profile = profile_name
            logger.info(f"Profil de risque mis à jour: {profile_name}")
            return True
        else:
            logger.error(f"Profil de risque inconnu: {profile_name}")
            return False
    
    def _calculate_score_factor(self, score: float) -> float:
        """
        Calcule un facteur de taille basé sur le score de l'opportunité
        
        Args:
            score: Score de l'opportunité (0-100)
            
        Returns:
            Facteur de taille (0.5-1.2)
        """
        # Convertir le score (0-100) en facteur (0.5-1.2)
        # 70 = 1.0 (neutre)
        # < 70 = réduit
        # > 70 = augmenté
        
        if score < 70:
            # Réduction linéaire pour les scores < 70
            # 50 -> 0.5, 60 -> 0.75, 70 -> 1.0
            return 0.5 + (score - 50) / 40
        else:
            # Augmentation pour les scores > 70
            # 70 -> 1.0, 85 -> 1.1, 100 -> 1.2
            return 1.0 + (score - 70) / 150
    
    def _volatility_adjustment(self, volatility: float) -> float:
        """
        Calcule un facteur d'ajustement basé sur la volatilité
        
        Args:
            volatility: Volatilité prédite (0-1)
            
        Returns:
            Facteur d'ajustement (0.5-1.5)
        """
        # Convertir la volatilité en facteur
        # Volatilité standard (0.03) = facteur 1.0
        standard_volatility = 0.03
        
        if volatility <= standard_volatility:
            # Volatilité faible -> positions plus grandes
            # 0.01 -> 1.5, 0.02 -> 1.25, 0.03 -> 1.0
            return 1.0 + ((standard_volatility - volatility) / standard_volatility) * 0.5
        else:
            # Volatilité élevée -> positions plus petites
            # 0.03 -> 1.0, 0.06 -> 0.5
            return max(0.5, 1.0 - ((volatility - standard_volatility) / standard_volatility) * 0.5)
    
    def _calculate_kelly_criterion(self) -> float:
        """
        Calcule la portion optimale du capital à risquer selon le critère de Kelly
        
        Returns:
            Fraction optimale du capital à risquer (0-1)
        """
        # Calculer la win rate sur l'historique récent
        if len(self.trade_history) < 10:
            return 0.5  # Valeur par défaut si pas assez de données
        
        # Récupérer les 20 derniers trades (ou moins si pas assez)
        recent_trades = self.trade_history[-20:]
        
        # Calculer le win rate
        winners = [t for t in recent_trades if t["pnl_percent"] > 0]
        win_rate = len(winners) / len(recent_trades)
        
        # Calculer le ratio gain/perte moyen
        if winners and len(recent_trades) > len(winners):
            avg_win = sum(t["pnl_percent"] for t in winners) / len(winners)
            losers = [t for t in recent_trades if t["pnl_percent"] <= 0]
            avg_loss = abs(sum(t["pnl_percent"] for t in losers) / len(losers))
            
            # Formule de Kelly: f* = (p*b - q) / b
            # où p = probabilité de gagner, q = probabilité de perdre, b = ratio gain/perte
            win_loss_ratio = avg_win / avg_loss if avg_loss > 0 else 1
            
            kelly_fraction = (win_rate * win_loss_ratio - (1 - win_rate)) / win_loss_ratio
            
            # Limiter la fraction entre 0.1 et 0.5 pour plus de sécurité
            kelly_fraction = max(0.1, min(0.5, kelly_fraction))
            
            return kelly_fraction
        
        return 0.25  # Valeur conservatrice par défaut
    
    def _adjust_risk_profile_based_on_performance(self) -> None:
        """
        Ajuste automatiquement le profil de risque en fonction des performances récentes
        """
        # Calculer la performance sur les derniers trades
        if len(self.trade_history) < 5:
            return  # Pas assez de données
        
        # Récupérer les 10 derniers trades (ou moins si pas assez)
        recent_trades = self.trade_history[-10:]
        
        # Calculer le PnL cumulé
        cumulative_pnl = sum(t["pnl_percent"] for t in recent_trades)
        
        # Calculer le win rate
        winners = len([t for t in recent_trades if t["pnl_percent"] > 0])
        win_rate = winners / len(recent_trades) * 100
        
        # Ajuster le profil en fonction de la performance
        if cumulative_pnl > 15 and win_rate > 60:
            # Performance très bonne -> profil agressif
            if self.current_risk_profile != "very_aggressive":
                self.set_risk_profile("very_aggressive")
                logger.info("Passage en mode très agressif basé sur la performance exceptionnelle")
        
        elif cumulative_pnl > 8 and win_rate > 55:
            # Bonne performance -> profil légèrement agressif
            if self.current_risk_profile != "aggressive" and self.current_risk_profile != "very_aggressive":
                self.set_risk_profile("aggressive")
                logger.info("Passage en mode agressif basé sur la bonne performance")
        
        elif cumulative_pnl < -10 or win_rate < 40:
            # Mauvaise performance -> profil conservateur
            if self.current_risk_profile != "conservative" and self.current_risk_profile != "defensive":
                self.set_risk_profile("conservative")
                logger.info("Passage en mode conservateur basé sur la performance médiocre")
        
        elif cumulative_pnl < -15 or win_rate < 30:
            # Très mauvaise performance -> profil défensif
            if self.current_risk_profile != "defensive":
                self.set_risk_profile("defensive")
                logger.info("Passage en mode défensif basé sur la performance très faible")
        
        else:
            # Performance moyenne -> profil équilibré
            if self.current_risk_profile not in ["balanced", "aggressive", "very_aggressive"]:
                self.set_risk_profile("balanced")
                logger.info("Retour au mode équilibré basé sur la performance stable")
    
    def _log_position_sizing(self, symbol: str, base_risk: float, 
                           final_size: float, factors: Dict) -> None:
        """
        Enregistre les détails de la décision de sizing pour l'analyse future
        
        Args:
            symbol: Paire de trading
            base_risk: Risque de base calculé
            final_size: Taille finale de la position
            factors: Facteurs qui ont influencé la décision
        """
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "symbol": symbol,
            "capital": self.current_capital,
            "base_risk_amount": base_risk,
            "final_position_size": final_size,
            "current_profile": self.current_risk_profile,
            "risk_capacity": self.risk_capacity,
            "consecutive_wins": self.consecutive_wins,
            "consecutive_losses": self.consecutive_losses,
            "factors": factors
        }
        
        self.risk_log.append(log_entry)
        
        # Limiter la taille du journal
        if len(self.risk_log) > 100:
            self.risk_log = self.risk_log[-100:]
        
        # Sauvegarder périodiquement
        if len(self.risk_log) % 10 == 0:
            self._save_risk_log()
    
    def _save_history(self) -> None:
        """Sauvegarde l'historique des trades et l'état actuel"""
        history_path = os.path.join(DATA_DIR, "risk_management", "trade_history.json")
        
        # Créer le répertoire si nécessaire
        os.makedirs(os.path.dirname(history_path), exist_ok=True)
        
        # Préparer les données à sauvegarder
        state_data = {
            "current_capital": self.current_capital,
            "risk_profile": self.current_risk_profile,
            "consecutive_wins": self.consecutive_wins,
            "consecutive_losses": self.consecutive_losses,
            "market_state": self.market_state,
            "risk_capacity": self.risk_capacity,
            "daily_risk_used": self.daily_risk_used,
            "last_risk_reset": self.last_risk_reset.isoformat(),
            "trade_history": self.trade_history,
            "updated_at": datetime.now().isoformat()
        }
        
        try:
            with open(history_path, 'w') as f:
                json.dump(state_data, f, indent=2, default=str)
        except Exception as e:
            logger.error(f"Erreur lors de la sauvegarde de l'historique des trades: {str(e)}")
    
    def _load_history(self) -> None:
        """Charge l'historique des trades et l'état précédent"""
        history_path = os.path.join(DATA_DIR, "risk_management", "trade_history.json")
        
        if not os.path.exists(history_path):
            return
        
        try:
            with open(history_path, 'r') as f:
                state_data = json.load(f)
            
            # Restaurer l'état
            self.current_capital = state_data.get("current_capital", self.initial_capital)
            self.current_risk_profile = state_data.get("risk_profile", "balanced")
            self.consecutive_wins = state_data.get("consecutive_wins", 0)
            self.consecutive_losses = state_data.get("consecutive_losses", 0)
            self.market_state = state_data.get("market_state", "normal")
            self.risk_capacity = state_data.get("risk_capacity", 1.0)
            self.daily_risk_used = state_data.get("daily_risk_used", 0.0)
            
            # Restaurer le timestamp du dernier reset
            try:
                self.last_risk_reset = datetime.fromisoformat(state_data.get("last_risk_reset", datetime.now().isoformat()))
            except ValueError:
                self.last_risk_reset = datetime.now()
            
            # Restaurer l'historique des trades
            self.trade_history = state_data.get("trade_history", [])
            
            logger.info(f"Historique des trades et état chargés: capital={self.current_capital}, profil={self.current_risk_profile}")
        except Exception as e:
            logger.error(f"Erreur lors du chargement de l'historique des trades: {str(e)}")
    
    def _save_risk_log(self) -> None:
        """Sauvegarde le journal des décisions de risque"""
        log_path = os.path.join(DATA_DIR, "risk_management", "sizing_decisions.json")
        
        # Créer le répertoire si nécessaire
        os.makedirs(os.path.dirname(log_path), exist_ok=True)
        
        try:
            with open(log_path, 'w') as f:
                json.dump(self.risk_log, f, indent=2, default=str)
        except Exception as e:
            logger.error(f"Erreur lors de la sauvegarde du journal des décisions: {str(e)}")
    
    def get_risk_profile_params(self) -> Dict:
        """
        Récupère les paramètres du profil de risque actuel
        
        Returns:
            Paramètres du profil actuel
        """
        return self.risk_profiles.get(self.current_risk_profile, self.risk_profiles["balanced"])
    
    def get_risk_parameters(self, symbol: str, lstm_prediction: Optional[Dict] = None) -> Dict:
        """
        Récupère tous les paramètres de risque actuels, possiblement ajustés par les prédictions LSTM
        
        Args:
            symbol: Paire de trading
            lstm_prediction: Prédictions du modèle LSTM (optionnel)
            
        Returns:
            Paramètres de risque complets
        """
        # Obtenir les paramètres de base du profil actuel
        profile = self.get_risk_profile_params()
        
        # Paramètres de base
        params = {
            "risk_per_trade_percent": profile["risk_per_trade_percent"],
            "stop_loss_percent": profile["stop_loss_percent"],
            "take_profit_percent": profile["take_profit_percent"],
            "leverage": profile["leverage"],
            "trailing_stop_activation": profile["trailing_stop_activation"],
            "trailing_stop_step": profile["trailing_stop_step"],
            "risk_profile": self.current_risk_profile,
            "risk_capacity": self.risk_capacity,
            "market_state": self.market_state
        }
        
        # Ajuster en fonction des prédictions LSTM si disponibles
        if lstm_prediction:
            # Chercher les prédictions de volatilité
            volatility_predicted = False
            
            for horizon_name, prediction in lstm_prediction.items():
                if horizon_name in ["3h", "4h", "short_term", "horizon_12"]:
                    volatility = prediction.get("predicted_volatility", None)
                    
                    if volatility is not None:
                        volatility_predicted = True
                        
                        # Ajuster le stop-loss et take-profit en fonction de la volatilité
                        volatility_percent = volatility * 100
                        std_volatility = 3.0  # 3% est considéré comme standard
                        
                        # Si volatilité élevée, élargir les stops; si faible, les resserrer
                        volatility_factor = volatility_percent / std_volatility
                        
                        params["stop_loss_percent"] = profile["stop_loss_percent"] * volatility_factor
                        params["take_profit_percent"] = profile["take_profit_percent"] * volatility_factor
                        
                        # S'assurer que les valeurs restent dans des limites raisonnables
                        params["stop_loss_percent"] = max(2.0, min(10.0, params["stop_loss_percent"]))
                        params["take_profit_percent"] = max(3.0, min(15.0, params["take_profit_percent"]))
                        
                        # Ajouter l'info de l'ajustement
                        params["volatility_adjustment_applied"] = True
                        params["predicted_volatility"] = volatility_percent
                        
                        break
            
            # Si pas de prédiction de volatilité, utiliser les paramètres standard
            if not volatility_predicted:
                params["volatility_adjustment_applied"] = False
        
        return params

def test_risk_manager():
    """Fonction de test simple pour vérifier l'initialisation du gestionnaire de risque"""
    risk_manager = AdaptiveRiskManager(
        initial_capital=200,
        risk_control_mode="balanced"
    )
    
    # Afficher les profils de risque
    print("Profils de risque configurés:")
    for profile_name, profile in risk_manager.risk_profiles.items():
        print(f"  {profile_name}:")
        for param, value in profile.items():
            print(f"    {param}: {value}")
    
    print(f"\nProfil actuel: {risk_manager.current_risk_profile}")
    print(f"Capacité de risque: {risk_manager.risk_capacity}")
    
    return risk_manager

if __name__ == "__main__":
    risk_manager = test_risk_manager()