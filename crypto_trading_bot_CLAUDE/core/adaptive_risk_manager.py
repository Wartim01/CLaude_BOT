# core/adaptive_risk_manager.py
"""
Gestionnaire de risque adaptatif basé sur l'IA
Ajuste la gestion du risque en fonction des prédictions du modèle LSTM
"""
import os
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Union, Tuple
from datetime import datetime, timedelta
import json
import math

from config.config import DATA_DIR, INITIAL_CAPITAL, MAX_DRAWDOWN_LIMIT
from config.trading_params import (
    RISK_PER_TRADE_PERCENT,
    MAX_CONCURRENT_TRADES,
    MAX_DAILY_TRADES,
    LEVERAGE,
    STOP_LOSS_PERCENT,
    TAKE_PROFIT_PERCENT
)
from ai.models.lstm_model import LSTMModel
from ai.models.feature_engineering import FeatureEngineering
from ai.market_anomaly_detector import MarketAnomalyDetector
from utils.logger import setup_logger

logger = setup_logger("adaptive_risk_manager")

class AdaptiveRiskManager:
    """
    Gère les risques et le capital du bot de trading de manière adaptative
    en utilisant les prédictions du modèle LSTM pour ajuster les paramètres
    """
    def __init__(self, initial_capital: float = INITIAL_CAPITAL, 
                model: Optional[LSTMModel] = None,
                feature_engineering: Optional[FeatureEngineering] = None,
                max_risk_per_trade: float = 10.0):
        """
        Initialise le gestionnaire de risque adaptatif
        
        Args:
            initial_capital: Capital initial
            model: Modèle LSTM pour les prédictions
            feature_engineering: Module d'ingénierie des caractéristiques
            max_risk_per_trade: Risque maximum par trade (%)
        """
        # Paramètres de base
        self.initial_capital = initial_capital
        self.available_balance = initial_capital
        self.equity = initial_capital
        self.model = model
        self.feature_engineering = feature_engineering or FeatureEngineering()
        self.max_risk_per_trade = max_risk_per_trade
        
        # Statistiques de trading
        self.daily_losses = 0
        self.daily_profits = 0
        self.positions_count = 0
        self.daily_trade_count = 0
        self.last_reset_date = datetime.now().date()
        self.consecutive_losses = 0
        self.win_streak = 0
        self.loss_streak = 0
        self.max_loss_streak = 0
        self.max_win_streak = 0
        self.current_drawdown = 0
        self.peak_equity = initial_capital
        
        # Paramètres adaptatifs
        self.risk_levels = {
            'aggressive': {
                'risk_per_trade': 5.0,
                'stop_loss': 3.0,
                'take_profit': 9.0,
                'leverage': LEVERAGE
            },
            'balanced': {
                'risk_per_trade': 3.0,
                'stop_loss': 4.0,
                'take_profit': 7.0,
                'leverage': LEVERAGE
            },
            'conservative': {
                'risk_per_trade': 2.0,
                'stop_loss': 5.0,
                'take_profit': 6.0,
                'leverage': LEVERAGE-1 if LEVERAGE > 1 else 1
            },
            'defensive': {
                'risk_per_trade': 1.0,
                'stop_loss': 6.0,
                'take_profit': 5.0,
                'leverage': max(1, LEVERAGE-2)
            }
        }
        
        # État de risque actuel
        self.current_risk_profile = 'balanced'
        
        # Détecteur d'anomalies de marché
        self.anomaly_detector = MarketAnomalyDetector(
            lookback_period=100,
            confidence_level=0.99,
            volatility_threshold=3.0,
            volume_threshold=5.0,
            price_gap_threshold=3.0,
            use_ml_models=True,
            model_dir=os.path.join(DATA_DIR, "models", "anomaly_detection")
        )
        
        # Historique des ajustements de risque
        self.risk_adjustments = []
        
        # Répertoire pour la sauvegarde des données
        self.data_dir = os.path.join(DATA_DIR, "risk_manager")
        os.makedirs(self.data_dir, exist_ok=True)
        
        # Modèle VaR pour la prédiction des risques extrêmes
        self.var_confidence_level = 0.95  # Niveau de confiance pour VaR (95%)
        self.var_window = 100  # Taille de la fenêtre pour le calcul de la VaR
        self.historical_returns = []  # Historique des rendements pour VaR
        
        # Chargement de l'état si disponible
        self._load_state()
    
    def update_account_balance(self, account_info: Dict) -> None:
        """
        Met à jour les informations de solde du compte
        
        Args:
            account_info: Informations du compte depuis l'API
        """
        # Extraire le solde USDT
        for asset in account_info.get("balances", []):
            if asset["asset"] == "USDT":
                self.available_balance = float(asset["free"])
                self.equity = float(asset["free"]) + float(asset["locked"])
                logger.info(f"Solde mis à jour: {self.available_balance} USDT disponible, {self.equity} USDT total")
                break
        
        # Mettre à jour le drawdown
        if self.equity > self.peak_equity:
            self.peak_equity = self.equity
            self.current_drawdown = 0
        else:
            self.current_drawdown = (self.peak_equity - self.equity) / self.peak_equity * 100
        
        # Réinitialiser les compteurs journaliers si nécessaire
        current_date = datetime.now().date()
        if current_date > self.last_reset_date:
            self.reset_daily_stats()
            self.last_reset_date = current_date
            
            # Ajuster le profil de risque en fonction de la performance
            self._adjust_risk_profile_daily()
    
    def reset_daily_stats(self) -> None:
        """
        Réinitialise les statistiques journalières
        """
        self.daily_losses = 0
        self.daily_profits = 0
        self.daily_trade_count = 0
        logger.info("Statistiques journalières réinitialisées")
    
    def update_position_stats(self, position_tracker) -> None:
        """
        Met à jour les statistiques de positions
        
        Args:
            position_tracker: Tracker de positions
        """
        self.positions_count = len(position_tracker.get_open_positions())
        self.daily_trade_count = position_tracker.get_daily_trades_count()
        
        # Calculer les profits/pertes journaliers
        today = datetime.now().date()
        
        daily_closed_positions = [
            p for p in position_tracker.get_closed_positions(limit=1000)
            if p.get("close_time") and p.get("close_time").date() == today
        ]
        
        self.daily_profits = sum([
            p.get("pnl_absolute", 0) for p in daily_closed_positions 
            if p.get("pnl_absolute", 0) > 0
        ])
        
        self.daily_losses = sum([
            p.get("pnl_absolute", 0) for p in daily_closed_positions 
            if p.get("pnl_absolute", 0) < 0
        ])
        
        logger.debug(f"Stats mises à jour: {self.positions_count} positions, {self.daily_trade_count} trades aujourd'hui")
    
    def can_open_new_position(self, position_tracker, market_data: Optional[Dict] = None) -> Dict:
        """
        Vérifie si une nouvelle position peut être ouverte avec vérification avancée
        des risques systémiques
        
        Args:
            position_tracker: Tracker de positions
            market_data: Données de marché actuelles (optionnel)
            
        Returns:
            Dict avec résultat et raison
        """
        # Mettre à jour les statistiques
        self.update_position_stats(position_tracker)
        
        # 1. Vérifications de base
        # Vérifier le nombre maximum de positions simultanées
        if self.positions_count >= MAX_CONCURRENT_TRADES:
            return {
                "can_open": False,
                "reason": f"Nombre maximum de positions simultanées atteint ({MAX_CONCURRENT_TRADES})"
            }
        
        # Vérifier le nombre maximum de trades par jour
        if self.daily_trade_count >= MAX_DAILY_TRADES:
            return {
                "can_open": False,
                "reason": f"Nombre maximum de trades journaliers atteint ({MAX_DAILY_TRADES})"
            }
        
        # Vérifier si le solde disponible est suffisant
        if self.available_balance <= 0:
            return {
                "can_open": False,
                "reason": "Solde insuffisant pour ouvrir une nouvelle position"
            }
        
        # 2. Vérifications avancées
        # Vérifier si nous sommes en drawdown critique
        if self.current_drawdown > MAX_DRAWDOWN_LIMIT:
            return {
                "can_open": False,
                "reason": f"Drawdown maximum dépassé ({self.current_drawdown:.2f}% > {MAX_DRAWDOWN_LIMIT}%)"
            }
        
        # Vérifier si nous avons trop de pertes consécutives
        if self.consecutive_losses >= 5:
            return {
                "can_open": False,
                "reason": f"Trop de pertes consécutives ({self.consecutive_losses})"
            }
        
        # Vérifier si les pertes journalières sont trop importantes
        daily_pnl = self.daily_profits + self.daily_losses
        max_daily_loss = -self.initial_capital * 0.05  # Max 5% de perte journalière
        
        if daily_pnl < max_daily_loss:
            return {
                "can_open": False,
                "reason": f"Perte journalière maximale atteinte ({daily_pnl:.2f} USDT)"
            }
        
        # 3. Vérification avancée des risques systémiques
        if market_data is not None:
            # Calculer la VaR (Value at Risk)
            if len(self.historical_returns) >= self.var_window:
                var = self._calculate_var()
                current_capital_at_risk = var * self.equity * self.positions_count
                
                # Si le risque est trop élevé, refuser de nouvelles positions
                max_capital_at_risk = self.equity * 0.2  # Max 20% du capital à risque
                
                if current_capital_at_risk > max_capital_at_risk:
                    return {
                        "can_open": False,
                        "reason": f"Risque systémique trop élevé (VaR: {var:.2f}%)"
                    }
        
        # 4. Vérifier les conditions de marché extrêmes
        extreme_conditions = self._detect_extreme_market_conditions(market_data)
        if extreme_conditions["detected"]:
            return {
                "can_open": False,
                "reason": f"Conditions de marché extrêmes détectées: {extreme_conditions['reason']}"
            }
        
        return {
            "can_open": True,
            "reason": "Conditions de risque respectées",
            "risk_profile": self.current_risk_profile
        }
    
    def calculate_position_size(self, symbol: str, opportunity: Dict, 
                              lstm_prediction: Optional[Dict] = None) -> float:
        """
        Calcule la taille de position optimale en fonction du risque adaptatif
        
        Args:
            symbol: Paire de trading
            opportunity: Opportunité de trading avec entrée et stop-loss
            lstm_prediction: Prédictions du modèle LSTM (optionnel)
            
        Returns:
            Quantité à trader
        """
        entry_price = opportunity.get("entry_price", 0)
        stop_loss_price = opportunity.get("stop_loss", 0)
        opportunity_score = opportunity.get("score", 50)
        
        if entry_price <= 0 or stop_loss_price <= 0:
            logger.error("Prix d'entrée ou de stop-loss invalide")
            return 0
        
        # Récupérer le profil de risque actuel
        risk_profile = self.risk_levels[self.current_risk_profile]
        
        # Ajuster le risque en fonction du score de l'opportunité
        base_risk = risk_profile["risk_per_trade"]
        adjusted_risk = self._adjust_risk_by_opportunity(base_risk, opportunity_score)
        
        # Si les prédictions LSTM sont disponibles, ajuster le risque
        if lstm_prediction is not None:
            adjusted_risk = self._adjust_risk_by_lstm_prediction(adjusted_risk, lstm_prediction)
        
        # Limiter le risque maximum
        adjusted_risk = min(adjusted_risk, self.max_risk_per_trade)
        
        # Calculer le risque en pourcentage (distance au stop-loss)
        if opportunity.get("side") == "BUY":
            risk_percent = (entry_price - stop_loss_price) / entry_price * 100
        else:
            risk_percent = (stop_loss_price - entry_price) / entry_price * 100
        
        if risk_percent <= 0:
            logger.error(f"Risque en pourcentage invalide: {risk_percent}%")
            return 0
        
        # Calculer le montant à risquer
        risk_amount = self.equity * (adjusted_risk / 100)
        
        # Ajuster le risque en fonction de l'historique récent
        if self.consecutive_losses > 0:
            # Réduire progressivement le risque après des pertes
            reduction_factor = max(0.5, 1.0 - (self.consecutive_losses * 0.1))
            risk_amount *= reduction_factor
            logger.info(f"Risque réduit de {(1-reduction_factor)*100:.0f}% après {self.consecutive_losses} pertes consécutives")
        
        # Calculer la taille de position en fonction du risque
        position_size = risk_amount / (risk_percent / 100 * entry_price)
        
        # Prendre en compte l'effet de levier
        leverage = risk_profile["leverage"]
        position_size = position_size * leverage
        
        # Limiter la taille de position au solde disponible
        max_position_size = self.available_balance * leverage / entry_price
        position_size = min(position_size, max_position_size)
        
        # Limiter la taille maximale de position à 25% du capital
        max_allowed_size = self.equity * 0.25 * leverage / entry_price
        position_size = min(position_size, max_allowed_size)
        
        # Arrondir la taille de position à la précision requise
        position_size = round(position_size, 5)
        
        logger.info(f"Taille de position calculée pour {symbol}: {position_size} ({risk_amount:.2f} USDT à risque)")
        
        return position_size
    
    def calculate_optimal_exit_levels(self, entry_price: float, side: str, 
                                    opportunity: Dict,
                                    lstm_prediction: Optional[Dict] = None) -> Dict:
        """
        Calcule les niveaux optimaux de stop-loss et take-profit basés sur les prédictions
        
        Args:
            entry_price: Prix d'entrée
            side: Direction (BUY/SELL)
            opportunity: Opportunité de trading
            lstm_prediction: Prédictions du modèle LSTM (optionnel)
            
        Returns:
            Dictionnaire avec les niveaux de sortie
        """
        # Récupérer le profil de risque actuel
        risk_profile = self.risk_levels[self.current_risk_profile]
        
        # Valeurs par défaut du profil de risque
        default_stop_percent = risk_profile["stop_loss"]
        default_tp_percent = risk_profile["take_profit"]
        
        # Ajuster en fonction du score de l'opportunité
        opportunity_score = opportunity.get("score", 50)
        
        # Plus le score est élevé, plus le ratio risque/récompense peut être agressif
        tp_multiplier = 1.0 + (opportunity_score - 50) / 100
        stop_multiplier = 1.0 - (opportunity_score - 50) / 200  # Moins sensible pour le stop
        
        stop_percent = default_stop_percent * stop_multiplier
        tp_percent = default_tp_percent * tp_multiplier
        
        # Si les prédictions LSTM sont disponibles, ajuster les niveaux
        if lstm_prediction is not None:
            # Utiliser la volatilité prédite pour ajuster les niveaux
            volatility_factor = 1.0
            
            # Pour chaque horizon, considérer la volatilité prédite
            for horizon, prediction in lstm_prediction.items():
                if "predicted_volatility" in prediction:
                    # Plus d'importance aux horizons courts
                    if "horizon_12" in horizon:  # Court terme
                        weight = 0.6
                    elif "horizon_24" in horizon:  # Moyen terme
                        weight = 0.3
                    else:  # Long terme
                        weight = 0.1
                    
                    # Normaliser la volatilité prédite
                    predicted_volatility = prediction["predicted_volatility"]
                    volatility_factor += predicted_volatility * weight
            
            # Ajuster les niveaux en fonction de la volatilité prédite
            volatility_factor = max(0.8, min(1.5, volatility_factor))
            stop_percent *= volatility_factor
            tp_percent *= volatility_factor
            
            # Utiliser le momentum prédit pour ajuster le ratio risque/récompense
            momentum_factor = 1.0
            
            for horizon, prediction in lstm_prediction.items():
                if "predicted_momentum" in prediction:
                    momentum = prediction["predicted_momentum"]
                    # Momentum positif = augmenter le take profit
                    if side == "BUY" and momentum > 0:
                        momentum_factor += momentum * 0.5
                    # Momentum négatif = augmenter le stop loss
                    elif side == "BUY" and momentum < 0:
                        stop_percent /= (1.0 + abs(momentum) * 0.3)
            
            tp_percent *= momentum_factor
        
        # Calculer les prix de stop-loss et take-profit
        if side == "BUY":
            stop_loss_price = entry_price * (1 - stop_percent / 100)
            take_profit_price = entry_price * (1 + tp_percent / 100)
            
            # NOUVEAU: Ajustement avancé des niveaux de stop-loss
            atr_value = opportunity.get("indicators", {}).get("atr", 0)
            if atr_value > 0:
                # Utiliser l'ATR pour définir un stop-loss plus intelligent
                atr_stop_level = entry_price - (atr_value * 1.5)
                # Prendre le maximum entre le stop % et le stop ATR
                stop_loss_price = max(stop_loss_price, atr_stop_level)
                
                # Vérifier les niveaux de support/résistance
                support_level = opportunity.get("indicators", {}).get("support_level", 0)
                if support_level > 0 and support_level < entry_price:
                    # Placer le stop légèrement sous le support
                    support_stop = support_level * 0.995
                    # Prendre le maximum entre les différents stops
                    stop_loss_price = max(stop_loss_price, support_stop)
        else:
            stop_loss_price = entry_price * (1 + stop_percent / 100)
            take_profit_price = entry_price * (1 - tp_percent / 100)
            
            # NOUVEAU: Ajustement avancé des niveaux de stop-loss
            atr_value = opportunity.get("indicators", {}).get("atr", 0)
            if atr_value > 0:
                # Utiliser l'ATR pour définir un stop-loss plus intelligent
                atr_stop_level = entry_price + (atr_value * 1.5)
                # Prendre le minimum entre le stop % et le stop ATR
                stop_loss_price = min(stop_loss_price, atr_stop_level)
                
                # Vérifier les niveaux de support/résistance
                resistance_level = opportunity.get("indicators", {}).get("resistance_level", 0)
                if resistance_level > 0 and resistance_level > entry_price:
                    # Placer le stop légèrement au-dessus de la résistance
                    resistance_stop = resistance_level * 1.005
                    # Prendre le minimum entre les différents stops
                    stop_loss_price = min(stop_loss_price, resistance_stop)
        
        return {
            "stop_loss_price": stop_loss_price,
            "take_profit_price": take_profit_price,
            "stop_loss_percent": stop_percent,
            "take_profit_percent": tp_percent
        }
    
    def update_after_trade_closed(self, trade_result: Dict) -> None:
        """
        Met à jour les statistiques après la fermeture d'un trade
        
        Args:
            trade_result: Résultat du trade
        """
        pnl = trade_result.get("pnl_absolute", 0)
        pnl_percent = trade_result.get("pnl_percent", 0)
        
        # Mettre à jour les métriques de base
        if pnl > 0:
            self.daily_profits += pnl
            self.consecutive_losses = 0
            self.win_streak += 1
            self.loss_streak = 0
            logger.info(f"Profit ajouté: {pnl} USDT (total journalier: {self.daily_profits} USDT)")
        else:
            self.daily_losses += pnl
            self.consecutive_losses += 1
            self.win_streak = 0
            self.loss_streak += 1
            logger.info(f"Perte ajoutée: {pnl} USDT (total journalier: {self.daily_losses} USDT)")
        
        # Mettre à jour les records
        self.max_win_streak = max(self.max_win_streak, self.win_streak)
        self.max_loss_streak = max(self.max_loss_streak, self.loss_streak)
        
        # Mettre à jour l'équité
        self.equity += pnl
        
        # Mettre à jour le peak equity et le drawdown
        if self.equity > self.peak_equity:
            self.peak_equity = self.equity
            self.current_drawdown = 0
        else:
            self.current_drawdown = (self.peak_equity - self.equity) / self.peak_equity * 100
        
        # Ajouter le rendement aux données historiques pour le calcul de la VaR
        if pnl_percent != 0:
            self.historical_returns.append(pnl_percent / 100)
            # Garder uniquement les dernières valeurs pour la VaR
            if len(self.historical_returns) > self.var_window * 2:
                self.historical_returns = self.historical_returns[-self.var_window * 2:]
        
        # Ajuster le profil de risque après chaque trade
        self._adjust_risk_profile_per_trade(pnl_percent)
        
        # Sauvegarder l'état
        self._save_state()
        
        # Si le drawdown dépasse 40%, déclencher une alerte
        if self.current_drawdown > 40:
            logger.warning(f"ALERTE: Drawdown élevé détecté ({self.current_drawdown:.2f}%)!")
    
    def get_risk_metrics(self) -> Dict:
        """
        Récupère les métriques de risque actuelles
        
        Returns:
            Métriques de risque
        """
        # Calculer la VaR si possible
        var = self._calculate_var() if len(self.historical_returns) >= self.var_window else None
        
        return {
            "initial_capital": self.initial_capital,
            "current_equity": self.equity,
            "available_balance": self.available_balance,
            "current_drawdown": self.current_drawdown,
            "peak_equity": self.peak_equity,
            "consecutive_losses": self.consecutive_losses,
            "win_streak": self.win_streak,
            "max_loss_streak": self.max_loss_streak,
            "max_win_streak": self.max_win_streak,
            "risk_profile": self.current_risk_profile,
            "risk_parameters": self.risk_levels[self.current_risk_profile],
            "value_at_risk": var,
            "var_confidence_level": self.var_confidence_level
        }
    
    def _adjust_risk_by_opportunity(self, base_risk: float, opportunity_score: float) -> float:
        """
        Ajuste le risque en fonction du score de l'opportunité
        
        Args:
            base_risk: Risque de base (%)
            opportunity_score: Score de l'opportunité (0-100)
            
        Returns:
            Risque ajusté (%)
        """
        # Normaliser le score (50 = neutre)
        score_factor = (opportunity_score - 50) / 50
        
        # Ajuster le risque (±30% max)
        adjustment = score_factor * 0.3
        adjusted_risk = base_risk * (1 + adjustment)
        
        return max(0.5, adjusted_risk)  # Minimum 0.5% de risque
    
    def _adjust_risk_by_lstm_prediction(self, base_risk: float, 
                                      lstm_prediction: Dict) -> float:
        """
        Ajuste le risque en fonction des prédictions du modèle LSTM
        
        Args:
            base_risk: Risque de base (%)
            lstm_prediction: Prédictions du modèle LSTM
            
        Returns:
            Risque ajusté (%)
        """
        # Facteur d'ajustement initial
        adjustment_factor = 1.0
        
        # 1. Ajuster en fonction de la confiance dans la direction prédite
        for horizon, prediction in lstm_prediction.items():
            if "direction_probability" in prediction:
                direction_prob = prediction["direction_probability"]
                
                # Confiance élevée = plus de risque, confiance faible = moins de risque
                confidence = abs(direction_prob - 0.5) * 2  # 0 à 1
                
                # Pondérer selon l'horizon
                if "horizon_12" in horizon:  # Court terme
                    weight = 0.5
                elif "horizon_24" in horizon:  # Moyen terme
                    weight = 0.3
                else:  # Long terme
                    weight = 0.2
                
                # Ajuster le facteur (±40% max basé sur la confiance)
                conf_adjustment = (confidence - 0.5) * 0.8 * weight
                adjustment_factor += conf_adjustment
        
        # 2. Ajuster en fonction de la volatilité prédite
        volatility_factor = 0
        
        for horizon, prediction in lstm_prediction.items():
            if "predicted_volatility" in prediction:
                volatility = prediction["predicted_volatility"]
                
                # Pondérer selon l'horizon
                if "horizon_12" in horizon:  # Court terme
                    weight = 0.6
                elif "horizon_24" in horizon:  # Moyen terme
                    weight = 0.3
                else:  # Long terme
                    weight = 0.1
                
                # Volatilité élevée = moins de risque
                # Normaliser autour de 1 (volatilité moyenne)
                if volatility > 1.2:
                    vol_adjustment = -0.2 * weight
                elif volatility < 0.8:
                    vol_adjustment = 0.2 * weight
                else:
                    vol_adjustment = 0
                
                volatility_factor += vol_adjustment
        
        adjustment_factor += volatility_factor
        
        # Limiter l'ajustement final
        adjustment_factor = max(0.6, min(1.4, adjustment_factor))
        
        return base_risk * adjustment_factor
    
    def _adjust_risk_profile_per_trade(self, pnl_percent: float) -> None:
        """
        Ajuste le profil de risque après chaque trade
        
        Args:
            pnl_percent: Pourcentage de profit/perte du trade
        """
        current_profile = self.current_risk_profile
        
        # Ajuster en fonction du résultat du trade
        if pnl_percent > 0:
            # Trade gagnant: devenir progressivement plus agressif
            if self.win_streak >= 3:
                if current_profile == 'conservative':
                    self._change_risk_profile('balanced')
                elif current_profile == 'balanced' and self.win_streak >= 5:
                    self._change_risk_profile('aggressive')
        else:
            # Trade perdant: devenir progressivement plus conservateur
            if self.loss_streak >= 2:
                if current_profile == 'aggressive':
                    self._change_risk_profile('balanced')
                elif current_profile == 'balanced' and self.loss_streak >= 3:
                    self._change_risk_profile('conservative')
                elif current_profile == 'conservative' and self.loss_streak >= 4:
                    self._change_risk_profile('defensive')
    
    def _adjust_risk_profile_daily(self) -> None:
        """
        Ajuste le profil de risque quotidiennement en fonction de la performance
        """
        # Calculer le P&L quotidien en pourcentage
        daily_pnl = self.daily_profits + self.daily_losses
        daily_pnl_percent = daily_pnl / self.equity * 100
        
        current_profile = self.current_risk_profile
        
        # Ajuster en fonction du P&L quotidien
        if daily_pnl_percent > 5:
            # Très bonne journée: devenir plus agressif
            if current_profile == 'defensive':
                self._change_risk_profile('conservative')
            elif current_profile == 'conservative':
                self._change_risk_profile('balanced')
            elif current_profile == 'balanced':
                self._change_risk_profile('aggressive')
        elif daily_pnl_percent < -3:
            # Mauvaise journée: devenir plus conservateur
            if current_profile == 'aggressive':
                self._change_risk_profile('balanced')
            elif current_profile == 'balanced':
                self._change_risk_profile('conservative')
            elif current_profile == 'conservative':
                self._change_risk_profile('defensive')
        
        # Ajuster en fonction du drawdown actuel
        if self.current_drawdown > 30:
            # Drawdown important: passer en mode défensif
            self._change_risk_profile('defensive')
        elif self.current_drawdown > 20:
            # Drawdown modéré: au plus conservateur
            if current_profile in ['aggressive', 'balanced']:
                self._change_risk_profile('conservative')
        elif self.current_drawdown < 10 and daily_pnl_percent > 0:
            # Faible drawdown et journée positive: monter d'un cran
            if current_profile == 'defensive':
                self._change_risk_profile('conservative')
            elif current_profile == 'conservative':
                self._change_risk_profile('balanced')
    
    def _change_risk_profile(self, new_profile: str) -> None:
        """
        Change le profil de risque actuel
        
        Args:
            new_profile: Nouveau profil de risque
        """
        if new_profile not in self.risk_levels:
            logger.error(f"Profil de risque invalide: {new_profile}")
            return
        
        if new_profile != self.current_risk_profile:
            old_profile = self.current_risk_profile
            self.current_risk_profile = new_profile
            
            # Enregistrer l'ajustement
            adjustment = {
                "timestamp": datetime.now().isoformat(),
                "old_profile": old_profile,
                "new_profile": new_profile,
                "reason": {
                    "drawdown": self.current_drawdown,
                    "consecutive_losses": self.consecutive_losses,
                    "win_streak": self.win_streak,
                    "equity": self.equity
                }
            }
            
            self.risk_adjustments.append(adjustment)
            logger.info(f"Profil de risque changé: {old_profile} -> {new_profile}")
    
    def _calculate_var(self, confidence_level: Optional[float] = None) -> float:
        """
        Calcule la Value at Risk (VaR) historique
        
        Args:
            confidence_level: Niveau de confiance (0-1)
            
        Returns:
            VaR en pourcentage du capital
        """
        if not self.historical_returns:
            return 0
        
        level = confidence_level or self.var_confidence_level
        
        # Calculer le quantile correspondant au niveau de confiance
        var_quantile = np.quantile(self.historical_returns, 1 - level)
        
        # VaR en pourcentage (valeur négative)
        return abs(var_quantile) * 100
    
    def _detect_extreme_market_conditions(self, market_data: Optional[Dict]) -> Dict:
        """
        Détecte les conditions de marché extrêmes en utilisant le détecteur d'anomalies
        
        Args:
            market_data: Données de marché
            
        Returns:
            Résultat de la détection
        """
        if not market_data:
            return {"detected": False}
        
        # Extraire les données OHLCV
        ohlcv = market_data.get("primary_timeframe", {}).get("ohlcv")
        if ohlcv is None or ohlcv.empty:
            return {"detected": False}
        
        # Prix actuel
        current_price = market_data.get("current_price")
        
        # Utiliser le détecteur d'anomalies
        result = self.anomaly_detector.detect_anomalies(
            data=ohlcv,
            current_price=current_price,
            return_details=True
        )
        
        # Si une anomalie est détectée, en informer le gestionnaire de risque
        if result["detected"]:
            logger.warning(f"Anomalie de marché détectée: {result['reason']}")
            
            # Si l'anomalie est très grave (score élevé), passer immédiatement en mode défensif
            if result.get("anomaly_count", 0) >= 3:
                self._change_risk_profile('defensive')
                logger.warning("Passage automatique en mode défensif en raison d'anomalies de marché multiples")
        
        return result
    
    def calculate_position_dynamic_stops(self, position_id: str, current_price: float, 
                                       position_data: Dict, 
                                       lstm_prediction: Optional[Dict] = None) -> Dict:
        """
        Calcule les niveaux de stop-loss dynamiques basés sur l'évolution du prix
        et les prédictions du modèle
        
        Args:
            position_id: Identifiant de la position
            current_price: Prix actuel
            position_data: Données de la position
            lstm_prediction: Prédictions du modèle LSTM
            
        Returns:
            Niveaux de stop-loss mis à jour
        """
        entry_price = position_data.get("entry_price", current_price)
        side = position_data.get("side", "BUY")
        current_stop = position_data.get("stop_loss_price", 0)
        take_profit = position_data.get("take_profit_price", 0)
        
        # Calculer le profit actuel en pourcentage
        if side == "BUY":
            current_profit_pct = (current_price - entry_price) / entry_price * 100
        else:
            current_profit_pct = (entry_price - current_price) / entry_price * 100
        
        # Trailing stop de base (basé sur ATR si disponible dans les prédictions)
        atr_value = None
        if lstm_prediction and "horizon_12" in lstm_prediction:
            volatility = lstm_prediction["horizon_12"].get("predicted_volatility", None)
            if volatility:
                # Estimer l'ATR à partir de la volatilité prédite
                atr_value = current_price * volatility / 100
        
        # Si pas d'ATR disponible, utiliser un pourcentage fixe
        if atr_value is None:
            # Pourcentage basé sur le profil de risque
            risk_params = self.risk_levels[self.current_risk_profile]
            trailing_pct = risk_params["stop_loss"] * 0.7  # 70% du stop-loss initial
            atr_value = current_price * trailing_pct / 100
        
        # NOUVEAU: Prendre en compte la prédiction de direction pour le trailing stop
        direction_confidence = 0.5  # Valeur neutre par défaut
        if lstm_prediction and "horizon_12" in lstm_prediction:
            direction_prob = lstm_prediction["horizon_12"].get("direction_probability", 0.5)
            
            # Si on est en position longue, on veut une proba > 0.5 pour être confiant
            # Si on est en position courte, on veut une proba < 0.5 pour être confiant
            if side == "BUY":
                direction_confidence = direction_prob
            else:
                direction_confidence = 1 - direction_prob
        
        # Ajuster le trailing stop en fonction de la confiance
        trailing_factor = 1.0
        if direction_confidence > 0.7:  # Forte confiance dans la continuation du mouvement
            trailing_factor = 1.3  # Plus lâche (plus de room)
        elif direction_confidence < 0.3:  # Faible confiance, risque de retournement
            trailing_factor = 0.7  # Plus serré (moins de room)
        
        # Calcul du nouveau stop-loss
        new_stop = current_stop
        
        if side == "BUY":
            # Pour les positions longues
            # Ne déplacer le stop que si le prix est en profit
            if current_profit_pct > 0:
                trailing_level = current_price - (atr_value * 2 * trailing_factor)
                
                # Si le trailing stop est au-dessus du stop actuel, mettre à jour
                if trailing_level > current_stop:
                    new_stop = trailing_level
                    
                    # Si profit > 3%, s'assurer que le stop est au minimum au point d'entrée (break-even)
                    if current_profit_pct > 3 and new_stop < entry_price:
                        new_stop = entry_price
                
                # NOUVEAU: Niveaux de sécurité progressifs basés sur le profit
                if current_profit_pct > 5:
                    # À 5% de profit, garantir au moins 50% du gain
                    min_stop = entry_price + (current_price - entry_price) * 0.5
                    new_stop = max(new_stop, min_stop)
                    
                if current_profit_pct > 10:
                    # À 10% de profit, garantir au moins 70% du gain
                    min_stop = entry_price + (current_price - entry_price) * 0.7
                    new_stop = max(new_stop, min_stop)
        else:
            # Pour les positions courtes
            if current_profit_pct > 0:
                trailing_level = current_price + (atr_value * 2 * trailing_factor)
                
                # Si le trailing stop est en-dessous du stop actuel, mettre à jour
                if trailing_level < current_stop:
                    new_stop = trailing_level
                    
                    # Si profit > 3%, s'assurer que le stop est au maximum au point d'entrée
                    if current_profit_pct > 3 and new_stop > entry_price:
                        new_stop = entry_price
                
                # NOUVEAU: Niveaux de sécurité progressifs basés sur le profit
                if current_profit_pct > 5:
                    # À 5% de profit, garantir au moins 50% du gain
                    min_stop = entry_price - (entry_price - current_price) * 0.5
                    new_stop = min(new_stop, min_stop)
                    
                if current_profit_pct > 10:
                    # À 10% de profit, garantir au moins 70% du gain
                    min_stop = entry_price - (entry_price - current_price) * 0.7
                    new_stop = min(new_stop, min_stop)
        
        # Si le LSTM prédit un fort mouvement dans la direction opposée, resserrer le stop
        if lstm_prediction and "horizon_12" in lstm_prediction:
            direction_prob = lstm_prediction["horizon_12"].get("direction_probability", 0.5)
            momentum = lstm_prediction["horizon_12"].get("predicted_momentum", 0)
            
            # Pour les longs, si forte probabilité de baisse, resserrer le stop
            if side == "BUY" and direction_prob < 0.3 and momentum < -0.3:
                tighter_stop = current_price - atr_value * 0.5  # Beaucoup plus serré
                if tighter_stop > new_stop:
                    new_stop = tighter_stop
                    logger.info(f"Stop-loss resserré en raison de prédiction baissière")
            
            # Pour les shorts, si forte probabilité de hausse, resserrer le stop
            elif side == "SELL" and direction_prob > 0.7 and momentum > 0.3:
                tighter_stop = current_price + atr_value * 0.5  # Beaucoup plus serré
                if tighter_stop < new_stop:
                    new_stop = tighter_stop
                    logger.info(f"Stop-loss resserré en raison de prédiction haussière")
        
        # NOUVEAU: Prendre en compte la détection d'anomalies
        # Si une anomalie est détectée, resserrer davantage le stop-loss pour protéger les profits
        if current_profit_pct > 2:  # Seulement si la position est en profit
            # Importer les données OHLCV récentes
            ohlcv = position_data.get("market_data", {}).get("ohlcv")
            
            if ohlcv is not None and not ohlcv.empty:
                # Détecter les anomalies
                anomaly_result = self.anomaly_detector.detect_anomalies(
                    data=ohlcv,
                    current_price=current_price,
                    return_details=False  # Version simplifiée pour être plus rapide
                )
                
                if anomaly_result.get("detected", False):
                    # Resserrer drastiquement le stop-loss
                    if side == "BUY":
                        emergency_stop = current_price - (atr_value * 0.5)  # Très serré
                        new_stop = max(new_stop, emergency_stop)
                    else:
                        emergency_stop = current_price + (atr_value * 0.5)  # Très serré
                        new_stop = min(new_stop, emergency_stop)
                    
                    logger.warning(f"Stop-loss d'urgence activé en raison d'anomalie de marché: {position_id}")
        
        return {
            "updated": new_stop != current_stop,
            "new_stop_loss": new_stop,
            "current_profit_pct": current_profit_pct,
            "atr_value": atr_value,
            "direction_confidence": direction_confidence,
            "trailing_factor": trailing_factor
        }
    
    def _save_state(self) -> None:
        """
        Sauvegarde l'état du gestionnaire de risque
        """
        state = {
            "equity": self.equity,
            "available_balance": self.available_balance,
            "peak_equity": self.peak_equity,
            "current_drawdown": self.current_drawdown,
            "consecutive_losses": self.consecutive_losses,
            "win_streak": self.win_streak,
            "loss_streak": self.loss_streak,
            "max_loss_streak": self.max_loss_streak,
            "max_win_streak": self.max_win_streak,
            "current_risk_profile": self.current_risk_profile,
            "risk_adjustments": self.risk_adjustments,
            "historical_returns": self.historical_returns,
            "last_updated": datetime.now().isoformat()
        }
        
        try:
            with open(os.path.join(self.data_dir, "risk_state.json"), 'w') as f:
                json.dump(state, f, indent=2, default=str)
            logger.debug("État du gestionnaire de risque sauvegardé")
        except Exception as e:
            logger.error(f"Erreur lors de la sauvegarde de l'état: {str(e)}")
    
    def _load_state(self) -> None:
        """
        Charge l'état du gestionnaire de risque
        """
        state_path = os.path.join(self.data_dir, "risk_state.json")
        if not os.path.exists(state_path):
            logger.info("Aucun état précédent trouvé")
            return
        
        try:
            with open(state_path, 'r') as f:
                state = json.load(f)
            
            # Restaurer l'état
            self.equity = state.get("equity", self.initial_capital)
            self.available_balance = state.get("available_balance", self.initial_capital)
            self.peak_equity = state.get("peak_equity", self.initial_capital)
            self.current_drawdown = state.get("current_drawdown", 0)
            self.consecutive_losses = state.get("consecutive_losses", 0)
            self.win_streak = state.get("win_streak", 0)
            self.loss_streak = state.get("loss_streak", 0)
            self.max_loss_streak = state.get("max_loss_streak", 0)
            self.max_win_streak = state.get("max_win_streak", 0)
            self.current_risk_profile = state.get("current_risk_profile", "balanced")
            self.risk_adjustments = state.get("risk_adjustments", [])
            self.historical_returns = state.get("historical_returns", [])
            
            logger.info(f"État du gestionnaire de risque chargé (capital: {self.equity} USDT)")
        except Exception as e:
            logger.error(f"Erreur lors du chargement de l'état: {str(e)}")
    
    def get_adaptive_parameters(self, symbol: str = None, prediction_data: Dict = None) -> Dict:
        """
        Retourne les paramètres adaptatifs actuels pour le trading
        
        Args:
            symbol: Paire de trading (optionnel)
            prediction_data: Données de prédiction (optionnel)
            
        Returns:
            Paramètres adaptatifs
        """
        # Récupérer les paramètres du profil de risque actuel
        risk_params = self.risk_levels[self.current_risk_profile]
        
        # Ajuster en fonction des prédictions si disponibles
        if prediction_data:
            # Calculer un facteur d'ajustement basé sur les prédictions
            volatility_factor = 1.0
            direction_confidence = 0.5
            
            # Extraire les prédictions du modèle LSTM
            if "lstm_prediction" in prediction_data:
                lstm_pred = prediction_data["lstm_prediction"]
                
                # Calculer la moyenne des volatilités prédites (pondérée par l'horizon)
                volatility_sum = 0
                volatility_weights = 0
                
                for horizon, pred in lstm_pred.items():
                    if "predicted_volatility" in pred:
                        # Pondération selon l'horizon
                        if "horizon_12" in horizon:  # Court terme
                            weight = 0.6
                        elif "horizon_24" in horizon:  # Moyen terme
                            weight = 0.3
                        else:  # Long terme
                            weight = 0.1
                        
                        volatility_sum += pred["predicted_volatility"] * weight
                        volatility_weights += weight
                
                if volatility_weights > 0:
                    avg_volatility = volatility_sum / volatility_weights
                    # Normaliser autour de 1.0
                    volatility_factor = avg_volatility
                
                # Calculer la confiance dans la direction
                if "horizon_12" in lstm_pred:  # Utiliser l'horizon court terme
                    direction_prob = lstm_pred["horizon_12"].get("direction_probability", 0.5)
                    direction_confidence = abs(direction_prob - 0.5) * 2  # 0 à 1
            
            # Ajuster les paramètres
            # Si volatilité élevée, augmenter le stop-loss et le take-profit
            stop_loss = risk_params["stop_loss"] * volatility_factor
            take_profit = risk_params["take_profit"] * volatility_factor
            
            # Si confiance élevée, augmenter le risque par trade
            risk_per_trade = risk_params["risk_per_trade"] * (1 + (direction_confidence - 0.5))
            
            return {
                "risk_per_trade_percent": risk_per_trade,
                "stop_loss_percent": stop_loss,
                "take_profit_percent": take_profit,
                "leverage": risk_params["leverage"],
                "risk_profile": self.current_risk_profile,
                "adjustment_factors": {
                    "volatility": volatility_factor,
                    "direction_confidence": direction_confidence
                }
            }
        
        # Sinon, retourner les paramètres standard
        return {
            "risk_per_trade_percent": risk_params["risk_per_trade"],
            "stop_loss_percent": risk_params["stop_loss"],
            "take_profit_percent": risk_params["take_profit"],
            "leverage": risk_params["leverage"],
            "risk_profile": self.current_risk_profile
        }

    # NOUVEAU: Méthodes pour la gestion avancée des risques

    def calculate_kelly_position_size(self, win_rate: float, reward_risk_ratio: float, 
                                    equity: float, max_risk_percent: float = 5.0) -> float:
        """
        Calcule la taille de position optimale avec le critère de Kelly
        
        Args:
            win_rate: Taux de gain (0-1)
            reward_risk_ratio: Ratio récompense/risque
            equity: Capital disponible
            max_risk_percent: Risque maximum autorisé (%)
            
        Returns:
            Taille optimale de position en pourcentage du capital
        """
        # Formule de Kelly: f* = (p*b - q)/b où p=win_rate, q=1-p, b=reward/risk
        kelly_fraction = (win_rate * reward_risk_ratio - (1 - win_rate)) / reward_risk_ratio
        
        # Limiter à la fraction maximale (généralement 1/2 Kelly est plus prudent)
        half_kelly = kelly_fraction / 2
        
        # Limiter au risque maximum autorisé
        risk_pct = min(max_risk_percent, half_kelly * 100)
        
        # Retourner la taille en montant
        return equity * (risk_pct / 100)

    def calculate_optimal_take_profit_levels(self, entry_price: float, side: str, 
                                         market_data: Dict, 
                                         prediction_data: Dict) -> List[Dict]:
        """
        Calcule plusieurs niveaux de take-profit optimaux en fonction des prédictions
        
        Args:
            entry_price: Prix d'entrée
            side: Direction (BUY/SELL)
            market_data: Données de marché
            prediction_data: Prédictions du modèle
            
        Returns:
            Liste de niveaux de take-profit avec pourcentages de sortie
        """
        # Récupérer la volatilité et le momentum prédits
        volatility = 0.03  # Valeur par défaut
        momentum = 0.0    # Valeur par défaut
        
        if "lstm_prediction" in prediction_data and "horizon_12" in prediction_data["lstm_prediction"]:
            pred = prediction_data["lstm_prediction"]["horizon_12"]
            volatility = pred.get("predicted_volatility", volatility)
            momentum = pred.get("predicted_momentum", momentum)
        
        # Calculer les niveaux de take-profit
        tp_levels = []
        
        # Niveau 1: Take-profit conservateur (33% de la position)
        if side == "BUY":
            tp1_percent = 2.0 + volatility * 100  # Base + ajustement pour volatilité
            tp1_price = entry_price * (1 + tp1_percent/100)
        else:
            tp1_percent = 2.0 + volatility * 100
            tp1_price = entry_price * (1 - tp1_percent/100)
        
        tp_levels.append({
            "price": tp1_price,
            "percent": tp1_percent,
            "portion": 0.33,  # 33% de la position
            "description": "Take-profit conservateur"
        })
        
        # Niveau 2: Take-profit modéré (33% de la position)
        if side == "BUY":
            tp2_percent = 5.0 + volatility * 200  # Base + ajustement pour volatilité
            tp2_price = entry_price * (1 + tp2_percent/100)
        else:
            tp2_percent = 5.0 + volatility * 200
            tp2_price = entry_price * (1 - tp2_percent/100)
        
        tp_levels.append({
            "price": tp2_price,
            "percent": tp2_percent,
            "portion": 0.33,
            "description": "Take-profit modéré"
        })
        
        # Niveau 3: Take-profit agressif (34% restant de la position)
        # Ajuster en fonction du momentum prédit
        momentum_factor = 1.0 + abs(momentum) * 2  # Plus de momentum = take-profit plus éloigné
        
        if side == "BUY":
            tp3_percent = 10.0 * momentum_factor
            tp3_price = entry_price * (1 + tp3_percent/100)
        else:
            tp3_percent = 10.0 * momentum_factor
            tp3_price = entry_price * (1 - tp3_percent/100)
        
        tp_levels.append({
            "price": tp3_price,
            "percent": tp3_percent,
            "portion": 0.34,
            "description": "Take-profit agressif"
        })
        
        return tp_levels

    def adjust_risk_for_correlated_positions(self, symbol: str, 
                                           correlated_symbols: Dict[str, float],
                                           position_tracker) -> float:
        """
        Ajuste le risque d'une nouvelle position en fonction des corrélations avec positions existantes
        
        Args:
            symbol: Symbole à trader
            correlated_symbols: Dictionnaire de symboles corrélés avec leurs coefficients
            position_tracker: Tracker de positions
            
        Returns:
            Facteur d'ajustement du risque (0-1)
        """
        # Récupérer les positions ouvertes
        open_positions = position_tracker.get_open_positions()
        
        # Si pas de positions ouvertes, pas d'ajustement
        if not open_positions:
            return 1.0
        
        # Calculer le risque corrélé
        correlated_risk = 0.0
        total_weight = 0.0
        
        for position in open_positions:
            position_symbol = position.get("symbol", "")
            
            # Si le symbole est dans la liste des corrélations
            if position_symbol in correlated_symbols:
                correlation = correlated_symbols[position_symbol]
                absolute_correlation = abs(correlation)
                
                # Plus la corrélation est forte, plus le poids est élevé
                weight = absolute_correlation * absolute_correlation  # Carré pour donner plus d'importance aux fortes corrélations
                
                # Accumulation des risques corrélés
                correlated_risk += weight
                total_weight += 1.0
        
        # Calcul du facteur d'ajustement
        # Plus le risque corrélé est élevé, plus le facteur est bas
        if total_weight > 0:
            # Normaliser entre 0.5 (réduction de 50%) et 1.0 (pas de réduction)
            reduction_factor = 1.0 - (correlated_risk / total_weight) * 0.5
            return max(0.5, reduction_factor)
        
        return 1.0  # Pas d'ajustement si pas de poids total