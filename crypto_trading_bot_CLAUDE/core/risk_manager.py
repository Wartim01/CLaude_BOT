# core/risk_manager.py
"""
Module de gestion des risques pour le bot de trading
Implémente des stratégies avancées d'allocation de capital et de gestion des risques
"""
import logging
import numpy as np
import pandas as pd
import os
from typing import Dict, List, Optional, Union
from datetime import datetime

from config.trading_params import (
    MAX_OPEN_POSITIONS, 
    MAX_POSITION_SIZE_USD,
    MAX_POSITION_SIZE_PERCENT,
    POSITION_SIZING_METHOD,
    MAX_ACCOUNT_RISK_PERCENT,
    MAX_DAILY_DRAWDOWN,
    DEFAULT_STOP_LOSS_PERCENT,
    TARGET_PROFIT_PERCENT
)
from utils.logger import setup_logger
from utils.correlation_matrix import CorrelationMatrix

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
logger = setup_logger("risk_manager")

class RiskManager:
    """
    Gère l'allocation de capital et les risques du portefeuille
    """
    def __init__(self):
        self.account_balance = 0.0
        self.available_balance = 0.0
        self.max_position_size_usd = MAX_POSITION_SIZE_USD
        self.max_positions = MAX_OPEN_POSITIONS
        self.daily_pnl = 0.0  # Profit/Loss quotidien
        self.initial_daily_balance = 0.0  # Solde au début de la journée
        self.daily_high_balance = 0.0  # Solde le plus élevé de la journée
        
        # Historique des performances pour ajuster dynamiquement les paramètres
        self.performance_history = []
        self.win_rate = 0.5  # Taux de réussite initial (50%)
        self.avg_win_loss_ratio = 1.5  # Ratio gain/perte initial
        
        # Paramètres de Kelly pour le sizing
        self.kelly_fraction = 0.3  # Fraction de Kelly pour limiter le risque
        
        # Variables pour le suivi des corrélations
        self.position_correlation = {}  # Corrélation entre les paires
        
        # Facteurs de risque de marché
        self.market_risk_factors = {
            "volatility": 1.0,  # Multiplicateur basé sur la volatilité du marché
            "trend_strength": 1.0,  # Multiplicateur basé sur la force de la tendance
            "liquidity": 1.0  # Multiplicateur basé sur la liquidité du marché
        }
        
        # Dernière mise à jour des limites
        self.last_limits_update = datetime.now()
        
        # Initialiser la matrice de corrélation
        self.correlation_matrix = CorrelationMatrix(
            cache_duration=3600,  # 1 heure de cache
            min_data_points=30    # Minimum 30 points pour calculer les corrélations
        )
        
        logger.info("Gestionnaire de risques initialisé")
    
    def update_account_balance(self, account_info: Dict) -> None:
        """
        Met à jour le solde du compte à partir des informations fournies
        
        Args:
            account_info: Informations sur le compte de trading
        """
        # Extraire le solde et le solde disponible
        if "totalWalletBalance" in account_info:
            self.account_balance = float(account_info["totalWalletBalance"])
        elif "totalBalance" in account_info:
            self.account_balance = float(account_info["totalBalance"])
        
        if "availableBalance" in account_info:
            self.available_balance = float(account_info["availableBalance"])
        else:
            # Estimation si non disponible
            self.available_balance = self.account_balance * 0.9
        
        # Mise à jour des métriques quotidiennes
        current_date = datetime.now().date()
        if not hasattr(self, "last_reset_date") or self.last_reset_date != current_date:
            self.initial_daily_balance = self.account_balance
            self.daily_high_balance = self.account_balance
            self.daily_pnl = 0.0
            self.last_reset_date = current_date
            logger.info(f"Métriques quotidiennes réinitialisées. Solde initial: {self.initial_daily_balance:.2f}")
        else:
            self.daily_pnl = self.account_balance - self.initial_daily_balance
            if self.account_balance > self.daily_high_balance:
                self.daily_high_balance = self.account_balance
        
        logger.info(f"Solde du compte mis à jour: {self.account_balance:.2f} (disponible: {self.available_balance:.2f})")
    
    def calculate_position_size(self, symbol: str, opportunity: Dict) -> float:
        """
        Calcule la taille optimale de la position en fonction de plusieurs facteurs:
        - Score de l'opportunité
        - Volatilité du marché
        - Risque de drawdown
        - Fraction de Kelly pour optimiser la croissance du capital
        
        Args:
            symbol: Symbole de la paire de trading
            opportunity: Détails de l'opportunité de trading
            
        Returns:
            Taille de la position en USD
        """
        # Vérifier si le solde est suffisant
        if self.available_balance <= 0:
            logger.warning(f"Solde disponible insuffisant: {self.available_balance}")
            return 0
        
        # Extraire les informations sur l'opportunité
        score = opportunity.get("score", 0)
        entry_price = opportunity.get("entry_price", 0)
        stop_loss = opportunity.get("stop_loss", 0)
        risk_reward_ratio = opportunity.get("risk_reward_ratio", 1.0)
        
        # Si les informations de prix sont manquantes, sortir
        if not entry_price or not stop_loss:
            logger.error(f"Informations de prix manquantes pour {symbol}")
            return 0
        
        # Calcul du risque par unité (en pourcentage)
        risk_per_unit_percent = abs((entry_price - stop_loss) / entry_price * 100)
        if risk_per_unit_percent <= 0:
            logger.error(f"Risque par unité invalide pour {symbol}: {risk_per_unit_percent}%")
            return 0
        
        # 1. Méthode de base: sizing par risque fixe
        if POSITION_SIZING_METHOD == "fixed_risk":
            # Pourcentage du compte à risquer (ajusté par le score)
            score_factor = min(1.0, score / 100)  # Score normalisé entre 0 et 1
            risk_percent = MAX_ACCOUNT_RISK_PERCENT * score_factor
            
            # Calcul du montant à risquer
            risk_amount = self.account_balance * (risk_percent / 100)
            
            # Calcul de la taille de la position
            position_size = risk_amount / (risk_per_unit_percent / 100)
            
        # 2. Méthode avancée: fraction de Kelly
        elif POSITION_SIZING_METHOD == "kelly":
            # Estimer la probabilité de succès à partir du score
            probability = 0.5 + (score - 50) * 0.005  # 50 → 0.5, 100 → 0.75
            probability = max(0.1, min(0.9, probability))  # Limiter entre 0.1 et 0.9
            
            # Calculer la fraction de Kelly
            if risk_reward_ratio > 0:
                kelly = probability - (1 - probability) / risk_reward_ratio
                kelly = max(0, kelly) * self.kelly_fraction  # Fraction de Kelly conservative
                
                # Appliquer la fraction au solde du compte
                risk_percent = kelly * 100
                risk_amount = self.account_balance * kelly
                
                # Calcul de la taille de la position
                position_size = risk_amount / (risk_per_unit_percent / 100)
            else:
                position_size = 0
        
        # 3. Méthode par défaut: sizing proportionnel au score
        else:
            # Pourcentage du compte à allouer basé sur le score
            allocation_percent = (MAX_POSITION_SIZE_PERCENT * score) / 100
            position_size = self.account_balance * (allocation_percent / 100)
        
        # Appliquer les ajustements de risque du marché
        position_size *= self.market_risk_factors["volatility"]
        position_size *= self.market_risk_factors["trend_strength"]
        position_size *= self.market_risk_factors["liquidity"]
        
        # Appliquer les limites
        position_size = min(position_size, self.max_position_size_usd, self.available_balance)
        
        # Vérifier si nous avons atteint le drawdown quotidien maximum
        daily_drawdown_percent = self._calculate_daily_drawdown()
        if daily_drawdown_percent >= MAX_DAILY_DRAWDOWN:
            logger.warning(f"Drawdown quotidien maximum atteint: {daily_drawdown_percent:.2f}%. Trading arrêté pour la journée.")
            return 0
        
        # Ajuster la taille en fonction du drawdown actuel
        if daily_drawdown_percent > 0:
            # Réduire progressivement la taille lorsque le drawdown augmente
            drawdown_factor = 1 - (daily_drawdown_percent / MAX_DAILY_DRAWDOWN)
            position_size *= drawdown_factor
        
        logger.info(f"Taille de position calculée pour {symbol}: {position_size:.2f} USD (score: {score}, risque: {risk_per_unit_percent:.2f}%)")
        return position_size
    
    def can_open_new_position(self, position_tracker) -> bool:
        """
        Détermine si une nouvelle position peut être ouverte
        en fonction des positions actuelles et des limites de risque
        
        Args:
            position_tracker: Gestionnaire de positions actuel
            
        Returns:
            True si une nouvelle position peut être ouverte, False sinon
        """
        # Vérifier le nombre de positions ouvertes
        open_positions = position_tracker.get_all_open_positions()
        total_positions = sum(len(positions) for positions in open_positions.values())
        
        if total_positions >= self.max_positions:
            logger.info(f"Nombre maximum de positions atteint: {total_positions}/{self.max_positions}")
            return False
        
        # Vérifier le solde disponible
        if self.available_balance < 20:
            logger.warning(f"Solde disponible insuffisant: {self.available_balance:.2f} USD")
            return False
        
        # Vérifier si nous avons atteint le drawdown quotidien maximum
        daily_drawdown_percent = self._calculate_daily_drawdown()
        if daily_drawdown_percent >= MAX_DAILY_DRAWDOWN:
            logger.warning(f"Drawdown quotidien maximum atteint: {daily_drawdown_percent:.2f}%. Trading arrêté pour la journée.")
            return False
        
        # Calculer l'exposition totale actuelle
        total_exposure = self._calculate_total_exposure(position_tracker)
        if total_exposure >= self.account_balance * 0.75:  # 75% du compte exposé
            logger.info(f"Exposition maximale atteinte: {total_exposure:.2f}/{self.account_balance:.2f} USD")
            return False
        
        return True
    
    def _calculate_daily_drawdown(self) -> float:
        """
        Calcule le drawdown quotidien en pourcentage
        
        Returns:
            Drawdown quotidien en pourcentage
        """
        if self.daily_high_balance <= 0:
            return 0
            
        current_drawdown = (self.daily_high_balance - self.account_balance) / self.daily_high_balance * 100
        return max(0, current_drawdown)
    
    def _calculate_total_exposure(self, position_tracker) -> float:
        """
        Calcule l'exposition totale actuelle
        
        Args:
            position_tracker: Gestionnaire de positions
            
        Returns:
            Exposition totale en USD
        """
        total_exposure = 0.0
        
        open_positions = position_tracker.get_all_open_positions()
        for symbol, positions in open_positions.items():
            for position in positions:
                position_value = position.get("entry_price", 0) * position.get("quantity", 0)
                total_exposure += position_value
        
        return total_exposure
    
    def update_market_risk_factors(self, market_data: Dict) -> None:
        """
        Met à jour les facteurs de risque du marché en fonction des données reçues
        
        Args:
            market_data: Données sur l'état du marché
        """
        # Mise à jour une fois par heure maximum
        current_time = datetime.now()
        if (current_time - self.last_limits_update).total_seconds() < 3600:
            return
            
        self.last_limits_update = current_time
        
        # Mise à jour du facteur de volatilité
        if "market_volatility" in market_data:
            volatility_level = market_data["market_volatility"]
            
            if volatility_level == "high":
                self.market_risk_factors["volatility"] = 0.7  # Réduire les positions en cas de forte volatilité
            elif volatility_level == "low":
                self.market_risk_factors["volatility"] = 1.2  # Augmenter légèrement en cas de faible volatilité
            else:
                self.market_risk_factors["volatility"] = 1.0  # Normal
        
        # Mise à jour du facteur de force de tendance
        if "trend_strength" in market_data:
            trend_strength = market_data["trend_strength"]
            
            # Adapter la taille des positions à la force de la tendance
            self.market_risk_factors["trend_strength"] = min(1.3, max(0.7, 0.5 + trend_strength / 2))
        
        # Mise à jour du facteur de liquidité
        if "liquidity" in market_data:
            liquidity = market_data["liquidity"]
            
            if liquidity == "low":
                self.market_risk_factors["liquidity"] = 0.8  # Réduire les positions sur les marchés peu liquides
            else:
                self.market_risk_factors["liquidity"] = 1.0  # Normal pour liquidité moyenne/haute
        
        logger.info(f"Facteurs de risque du marché mis à jour: {self.market_risk_factors}")
    
    def update_performance_metrics(self, trade_result: Dict) -> None:
        """
        Met à jour les métriques de performance pour affiner la stratégie de gestion des risques
        
        Args:
            trade_result: Résultat d'un trade
        """
        # Ajouter le résultat à l'historique
        self.performance_history.append(trade_result)
        
        # Garder seulement les 100 derniers trades
        if len(self.performance_history) > 100:
            self.performance_history = self.performance_history[-100:]
        
        # Calculer le win rate
        wins = sum(1 for trade in self.performance_history if trade.get("pnl_percent", 0) > 0)
        self.win_rate = wins / len(self.performance_history) if self.performance_history else 0.5
        
        # Calculer le ratio gain/perte moyen
        wins = [trade.get("pnl_percent", 0) for trade in self.performance_history if trade.get("pnl_percent", 0) > 0]
        losses = [trade.get("pnl_percent", 0) for trade in self.performance_history if trade.get("pnl_percent", 0) < 0]
        
        avg_win = np.mean(wins) if wins else 0
        avg_loss = np.mean(losses) if losses else 0
        
        self.avg_win_loss_ratio = abs(avg_win / avg_loss) if avg_loss != 0 else 1.5
        
        # Ajuster la fraction de Kelly en fonction des performances récentes
        self._adjust_kelly_fraction()
        
        logger.info(f"Métriques de performance mises à jour: Win rate={self.win_rate:.2f}, Win/Loss ratio={self.avg_win_loss_ratio:.2f}")
    
    def _adjust_kelly_fraction(self) -> None:
        """
        Ajuste la fraction de Kelly en fonction des performances récentes
        """
        # Calculer la fraction de Kelly théorique
        if self.win_rate > 0 and self.avg_win_loss_ratio > 0:
            kelly = self.win_rate - ((1 - self.win_rate) / self.avg_win_loss_ratio)
            kelly = max(0, kelly)
            
            # Appliquer un facteur de sécurité pour être plus conservateur
            if self.win_rate < 0.45:  # Win rate faible
                safety_factor = 0.2
            elif self.win_rate < 0.55:  # Win rate moyen
                safety_factor = 0.3
            else:  # Win rate élevé
                safety_factor = 0.4
                
            self.kelly_fraction = kelly * safety_factor
            
            # Limiter la fraction de Kelly
            self.kelly_fraction = min(0.5, max(0.05, self.kelly_fraction))
            
            logger.debug(f"Fraction de Kelly ajustée à {self.kelly_fraction:.2f}")
    
    def calculate_dynamic_stop_loss(self, symbol: str, entry_price: float, direction: str, 
                                 market_state: Dict, volatility: float = None) -> Dict:
        """
        Calcule dynamiquement les niveaux de stop-loss et take-profit 
        en fonction de la volatilité du marché et d'autres facteurs
        
        Args:
            symbol: Paire de trading
            entry_price: Prix d'entrée
            direction: Direction du trade (BUY/SELL)
            market_state: État du marché
            volatility: Mesure de volatilité (ATR ou similaire)
            
        Returns:
            Dict avec stop_loss et take_profit
        """
        # Calculer le pourcentage de stop-loss de base
        is_long = direction == "BUY"
        base_sl_percent = DEFAULT_STOP_LOSS_PERCENT
        
        # Si la volatilité est fournie, l'utiliser pour ajuster le stop-loss
        if volatility is not None:
            # Volatilité normalisée (supposons que 2% est la volatilité moyenne)
            volatility_factor = volatility / 2.0
            
            # Ajuster le stop-loss en fonction de la volatilité
            adjusted_sl_percent = base_sl_percent * volatility_factor
            
            # Limiter l'ajustement pour éviter des stop-loss trop larges ou trop serrés
            adjusted_sl_percent = min(10.0, max(1.0, adjusted_sl_percent))
        else:
            # Utiliser les informations d'état du marché pour estimer la volatilité
            volatility_level = market_state.get("volatility", "medium")
            
            if volatility_level == "high":
                adjusted_sl_percent = base_sl_percent * 1.5
            elif volatility_level == "low":
                adjusted_sl_percent = base_sl_percent * 0.8
            else:
                adjusted_sl_percent = base_sl_percent
        
        # Calculer le pourcentage de take-profit en fonction du ratio de récompense cible
        target_rr_ratio = 2.0  # Ratio risque/récompense cible
        adjusted_tp_percent = adjusted_sl_percent * target_rr_ratio
        
        # Calculer les prix de stop-loss et take-profit
        if is_long:
            stop_loss = entry_price * (1 - adjusted_sl_percent / 100)
            take_profit = entry_price * (1 + adjusted_tp_percent / 100)
        else:
            stop_loss = entry_price * (1 + adjusted_sl_percent / 100)
            take_profit = entry_price * (1 - adjusted_tp_percent / 100)
        
        # Calculer les paramètres pour le trailing stop
        trailing_activation = entry_price * (1 + (TARGET_PROFIT_PERCENT * 0.3 / 100)) if is_long else entry_price * (1 - (TARGET_PROFIT_PERCENT * 0.3 / 100))
        trailing_callback = adjusted_sl_percent * 0.5 / 100  # % de rappel du trailing
        
        result = {
            "stop_loss": stop_loss,
            "take_profit": take_profit,
            "stop_loss_percent": adjusted_sl_percent,
            "take_profit_percent": adjusted_tp_percent,
            "risk_reward_ratio": target_rr_ratio,
            "trailing_activation": trailing_activation,
            "trailing_callback": trailing_callback
        }
        
        logger.info(f"Stop-loss dynamique calculé pour {symbol}: SL={adjusted_sl_percent:.2f}%, TP={adjusted_tp_percent:.2f}%, RR={target_rr_ratio}")
        return result
    
    def analyze_position_correlation(self, positions: Dict, market_data: Dict) -> Dict:
        """
        Analyse la corrélation entre les positions ouvertes pour identifier
        les concentrations de risque et ajuster les tailles de position en conséquence
        
        Args:
            positions: Positions actuellement ouvertes
            market_data: Données de marché pour les paires concernées
            
        Returns:
            Analyse de corrélation et recommandations
        """
        # Si moins de 2 positions, pas besoin d'analyse de corrélation
        if len(positions) < 2:
            return {"correlation": 0, "risk_concentration": "low", "recommendation": None}
        
        # Extraire les symboles et les poids des positions
        position_weights = {}
        for symbol, pos_list in positions.items():
            # Calculer la valeur totale des positions pour ce symbole
            total_value = sum(
                p.get("entry_price", 0) * p.get("quantity", 0)
                for p in pos_list if isinstance(p, dict)
            )
            position_weights[symbol] = total_value
        
        # Utiliser la matrice de corrélation pour analyser le portefeuille
        try:
            # Pour chaque fenêtre temporelle, calculer les métriques de risque
            short_term = self.correlation_matrix.calculate_portfolio_risk(
                position_weights, time_window='1d'
            )
            
            medium_term = self.correlation_matrix.calculate_portfolio_risk(
                position_weights, time_window='7d'
            )
            
            # Obtenir les paires hautement corrélées
            high_corr_pairs = self.correlation_matrix.get_highly_correlated_pairs(
                threshold=0.7, time_window='7d'
            )
            
            # Préparer le résultat
            result = {
                "short_term": short_term,
                "medium_term": medium_term,
                "high_corr_pairs": high_corr_pairs
            }
            
            logger.info(f"Analyse de corrélation: concentration de risque {medium_term['risk_concentration']} (corr. moyenne: {medium_term['average_correlation']:.2f})")
            return result
        except Exception as e:
            logger.error(f"Erreur lors de l'analyse de corrélation: {str(e)}")
            return {"correlation": 0, "risk_concentration": "error", "recommendation": None}
    
    def update_correlation_data(self, market_data_dict: Dict[str, pd.DataFrame], time_windows: List[str] = ['1d', '7d', '30d']) -> None:
        """
        Met à jour les données de corrélation entre les actifs
        
        Args:
            market_data_dict: Dictionnaire {symbol: DataFrame avec les prix}
            time_windows: Liste des fenêtres temporelles à calculer
        """
        try:
            for time_window in time_windows:
                # Mise à jour de la matrice de corrélation pour chaque fenêtre temporelle
                matrix = self.correlation_matrix.update_matrix(market_data_dict, time_window)
                
                logger.info(f"Matrice de corrélation {time_window} mise à jour: {len(matrix)} actifs")
                
                # Générer un rapport périodique (par exemple pour '7d')
                if time_window == '7d':
                    report_path = os.path.join(DATA_DIR, "reports", f"correlation_report_{datetime.now().strftime('%Y%m%d')}.json")
                    os.makedirs(os.path.dirname(report_path), exist_ok=True)
                    
                    report = self.correlation_matrix.generate_correlation_report(
                        time_window=time_window,
                        save_path=report_path
                    )
                    
                    # Générer une visualisation
                    viz_path = os.path.join(DATA_DIR, "reports", f"correlation_matrix_{datetime.now().strftime('%Y%m%d')}.png")
                    self.correlation_matrix.visualize_matrix(
                        time_window=time_window,
                        save_path=viz_path,
                        show_plot=False
                    )
        except Exception as e:
            logger.error(f"Erreur lors de la mise à jour des corrélations: {str(e)}")