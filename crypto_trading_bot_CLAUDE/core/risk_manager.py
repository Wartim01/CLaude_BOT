# core/risk_manager.py
"""
Gestionnaire de risques pour le bot de trading
"""
from typing import Dict, Optional
import logging
from datetime import datetime, timedelta

from config.config import INITIAL_CAPITAL
from config.trading_params import (
    RISK_PER_TRADE_PERCENT,
    MAX_CONCURRENT_TRADES,
    MAX_DAILY_TRADES,
    LEVERAGE,
    MAX_DRAWDOWN_LIMIT
)
from utils.logger import setup_logger

logger = setup_logger("risk_manager")

class RiskManager:
    """
    Gère les risques et le capital du bot de trading
    """
    def __init__(self):
        self.initial_capital = INITIAL_CAPITAL
        self.available_balance = INITIAL_CAPITAL
        self.equity = INITIAL_CAPITAL
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
        self.peak_equity = INITIAL_CAPITAL
        
        # NOUVEAU: Historique des performances
        self.performance_history = []
        self.volatility_history = []

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
        
        # Réinitialiser les compteurs journaliers si nécessaire
        current_date = datetime.now().date()
        if current_date > self.last_reset_date:
            self.reset_daily_stats()
            self.last_reset_date = current_date
    
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
    
    def can_open_new_position(self, position_tracker) -> bool:
        """
        Vérifie si une nouvelle position peut être ouverte
        
        Args:
            position_tracker: Tracker de positions
            
        Returns:
            True si une nouvelle position peut être ouverte, False sinon
        """
        # Mettre à jour les statistiques
        self.update_position_stats(position_tracker)
        
        # Vérifier le nombre maximum de positions simultanées
        if self.positions_count >= MAX_CONCURRENT_TRADES:
            logger.info(f"Nombre maximum de positions simultanées atteint ({MAX_CONCURRENT_TRADES})")
            return False
        
        # Vérifier le nombre maximum de trades par jour
        if self.daily_trade_count >= MAX_DAILY_TRADES:
            logger.info(f"Nombre maximum de trades journaliers atteint ({MAX_DAILY_TRADES})")
            return False
        
        # Vérifier si le solde disponible est suffisant
        if self.available_balance <= 0:
            logger.info("Solde insuffisant pour ouvrir une nouvelle position")
            return False
        
        return True
    
    def calculate_position_size(self, symbol: str, opportunity: Dict) -> float:
        """
        Calcule la taille de position optimale en fonction du risque - Version adaptative
        
        Args:
            symbol: Paire de trading
            opportunity: Opportunité de trading avec entrée et stop-loss
            
        Returns:
            Quantité à trader
        """
        entry_price = opportunity.get("entry_price", 0)
        stop_loss_price = opportunity.get("stop_loss", 0)
        score = opportunity.get("score", 50)
        
        if entry_price <= 0 or stop_loss_price <= 0:
            logger.error("Prix d'entrée ou de stop-loss invalide")
            return 0
        
        # Calculer le risque en pourcentage
        if opportunity.get("side") == "BUY":
            risk_percent = (entry_price - stop_loss_price) / entry_price * 100
        else:
            risk_percent = (stop_loss_price - entry_price) / entry_price * 100
        
        if risk_percent <= 0:
            logger.error(f"Risque en pourcentage invalide: {risk_percent}%")
            return 0
        
        # NOUVEAU: Risque adaptatif basé sur:
        # 1. Score de l'opportunité
        # 2. Historique récent de pertes/gains
        # 3. Drawdown actuel
        base_risk = RISK_PER_TRADE_PERCENT
        
        # Ajuster le risque en fonction du score
        score_factor = self._calculate_score_factor(score)
        
        # Ajuster le risque en fonction des pertes consécutives
        streak_factor = self._calculate_streak_factor()
        
        # Ajuster le risque en fonction du drawdown actuel
        drawdown_factor = self._calculate_drawdown_factor()
        
        # Calculer le facteur de risque combiné
        risk_factor = score_factor * streak_factor * drawdown_factor
        
        # Limite pour éviter des risques trop extrêmes
        risk_factor = max(0.3, min(1.5, risk_factor))
        
        # Risque final ajusté
        adjusted_risk = base_risk * risk_factor
        logger.info(f"Risque ajusté: {adjusted_risk:.2f}% (base: {base_risk}%, facteur: {risk_factor:.2f})")
        
        # Calculer le montant à risquer
        risk_amount = self.equity * (adjusted_risk / 100)
        
        # Calculer la taille de position en fonction du risque
        position_size = risk_amount / (risk_percent / 100 * entry_price)
        
        # Prendre en compte l'effet de levier
        position_size = position_size * LEVERAGE
        
        # Limiter la taille de position au solde disponible
        max_position_size = self.available_balance * LEVERAGE / entry_price
        position_size = min(position_size, max_position_size)
        
        # NOUVEAU: Limiter la taille maximale de position à 25% du capital total, quelle que soit la situation
        max_allowed_size = self.equity * 0.25 * LEVERAGE / entry_price
        position_size = min(position_size, max_allowed_size)
        
        # Arrondir la taille de position à la précision requise
        position_size = round(position_size, 5)
        
        logger.info(f"Taille de position calculée pour {symbol}: {position_size} (risque: {risk_amount:.2f} USDT)")
        return position_size
    
    def update_after_trade_closed(self, trade_result: Dict) -> None:
        """
        Met à jour les statistiques après la fermeture d'un trade
        
        Args:
            trade_result: Résultat du trade
        """
        pnl = trade_result.get("pnl_absolute", 0)
        
        if pnl > 0:
            self.daily_profits += pnl
            logger.info(f"Profit ajouté: {pnl} USDT (total journalier: {self.daily_profits} USDT)")
        else:
            self.daily_losses += pnl
            logger.info(f"Perte ajoutée: {pnl} USDT (total journalier: {self.daily_losses} USDT)")
    
    def get_risk_metrics(self) -> Dict:
        """
        Récupère les métriques de risque actuelles
        
        Returns:
            Métriques de risque
        """
        return {
            "initial_capital": self.initial_capital,
            "current_equity": self.equity,
            "available_balance": self.available_balance,
            "positions_count": self.positions_count,
            "daily_trade_count": self.daily_trade_count,
            "daily_profits": self.daily_profits,
            "daily_losses": self.daily_losses,
            "net_daily_pnl": self.daily_profits + self.daily_losses,
            "daily_roi_percent": (self.daily_profits + self.daily_losses) / self.initial_capital * 100
        }
    
    def calculate_adaptive_risk(self, symbol: str, opportunity: Dict, market_volatility: float) -> float:
        """
        Calcule le risque par trade de manière adaptative en fonction de la volatilité du marché
        
        Args:
            symbol: Paire de trading
            opportunity: Opportunité de trading
            market_volatility: Niveau de volatilité du marché (0-1)
            
        Returns:
            Pourcentage du capital à risquer
        """
        base_risk = RISK_PER_TRADE_PERCENT
        
        # Réduire le risque quand la volatilité est élevée
        if market_volatility > 0.7:  # Volatilité élevée
            return base_risk * 0.7
        elif market_volatility > 0.4:  # Volatilité moyenne
            return base_risk * 0.85
        else:  # Volatilité faible
            return base_risk * 1.1  # Plus agressif quand le marché est calme
    
    def _calculate_score_factor(self, score: int) -> float:
        """
        Calcule un facteur de risque basé sur le score de l'opportunité
        """
        # Plus le score est élevé, plus nous sommes confiants, donc nous prenons plus de risque
        if score >= 90:
            return 1.3  # Très confiant - augmenter le risque de 30%
        elif score >= 80:
            return 1.2
        elif score >= 70:
            return 1.1
        elif score >= 60:
            return 1.0  # Score normal - risque standard
        elif score >= 50:
            return 0.8
        else:
            return 0.6  # Score faible - réduire le risque de 40%
    
    def _calculate_streak_factor(self) -> float:
        """
        Calcule un facteur de risque basé sur les séquences de pertes/gains
        """
        # Après des pertes consécutives, réduire progressivement le risque
        if self.consecutive_losses >= 4:
            return 0.5  # Réduire le risque de 50% après 4 pertes consécutives
        elif self.consecutive_losses >= 3:
            return 0.6
        elif self.consecutive_losses >= 2:
            return 0.7
        elif self.consecutive_losses >= 1:
            return 0.8
        
        # Après des gains consécutifs, augmenter légèrement le risque
        if self.win_streak >= 3:
            return 1.2  # Augmenter le risque de 20% après 3 gains consécutifs
        elif self.win_streak >= 2:
            return 1.1
        
        return 1.0  # Pas d'ajustement
    
    def _calculate_drawdown_factor(self) -> float:
        """
        Calcule un facteur de risque basé sur le drawdown actuel
        """
        # Réduire le risque si nous sommes en drawdown significatif
        if self.current_drawdown > 35:
            return 0.5  # Réduire le risque de 50% si drawdown > 35%
        elif self.current_drawdown > 25:
            return 0.6
        elif self.current_drawdown > 15:
            return 0.8
        
        return 1.0  # Pas d'ajustement
    
    def update_after_trade_closed(self, trade_result: Dict) -> None:
        """
        Met à jour les statistiques après la fermeture d'un trade - Version améliorée
        
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
        
        # Enregistrer les données de performance
        self.performance_history.append({
            "timestamp": datetime.now().isoformat(),
            "trade_id": trade_result.get("trade_id"),
            "pnl": pnl,
            "pnl_percent": pnl_percent,
            "equity": self.equity,
            "drawdown": self.current_drawdown
        })
        
        # Si le drawdown dépasse 40%, déclencher une alerte
        if self.current_drawdown > 40:
            logger.warning(f"ALERTE: Drawdown élevé détecté ({self.current_drawdown:.2f}%)!")
            
            # Vous pourriez ajouter ici une logique pour arrêter temporairement le trading
            # ou réduire davantage la taille des positions
    
    def can_open_new_position(self, position_tracker) -> Dict:
        """
        Vérifie si une nouvelle position peut être ouverte - Version améliorée
        
        Args:
            position_tracker: Tracker de positions
            
        Returns:
            Dict avec résultat et raison
        """
        # Mettre à jour les statistiques
        self.update_position_stats(position_tracker)
        
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
        
        # NOUVEAU: Vérifier si nous sommes en drawdown critique
        if self.current_drawdown > MAX_DRAWDOWN_LIMIT:
            return {
                "can_open": False,
                "reason": f"Drawdown maximum dépassé ({self.current_drawdown:.2f}% > {MAX_DRAWDOWN_LIMIT}%)"
            }
        
        # NOUVEAU: Vérifier si nous avons trop de pertes consécutives
        if self.consecutive_losses >= 5:
            return {
                "can_open": False,
                "reason": f"Trop de pertes consécutives ({self.consecutive_losses})"
            }
        
        return {
            "can_open": True,
            "reason": "Conditions de risque respectées"
        }