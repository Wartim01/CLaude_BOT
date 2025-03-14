# strategies/strategy_base.py
"""
Classe de base pour les stratégies de trading
"""
import os
import json
import logging
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Union
from datetime import datetime

from config.config import DATA_DIR
from config.trading_params import MINIMUM_SCORE_TO_TRADE
from utils.logger import setup_logger

logger = setup_logger("strategy_base")

class StrategyBase(ABC):
    """
    Classe de base pour définir l'interface des stratégies de trading.
    Les stratégies concrètes doivent implémenter la méthode generate_signal.
    """
    def __init__(self, data_fetcher, market_analyzer, scoring_engine):
        self.data_fetcher = data_fetcher
        self.market_analyzer = market_analyzer
        self.scoring_engine = scoring_engine
        self.min_score = MINIMUM_SCORE_TO_TRADE
        
        # Répertoire pour les journaux de trades
        self.trades_dir = os.path.join(DATA_DIR, "trade_logs")
        if not os.path.exists(self.trades_dir):
            os.makedirs(self.trades_dir)
    
    @abstractmethod
    def generate_signal(self, symbol: str, data) -> dict:
        """
        Génère un signal de trading pour le symbole donné à partir des données fournies.
        
        Args:
            symbol: Le symbole du marché.
            data: Données de marché ou DataFrame d'indicateurs.
            
        Returns:
            Dictionnaire contenant au minimum la clé 'signal', éventuellement d'autres informations.
        """
        pass

    def evaluate_performance(self, signals: list) -> float:
        """
        Méthode utilitaire pour évaluer la performance d'une stratégie.
        
        Args:
            signals: Liste de signaux générés.
            
        Returns:
            Score de performance (pourcentage ou autre métrique).
        """
        # ...existing code or custom evaluation logic...
        return 0.0
    
    @abstractmethod
    def find_trading_opportunity(self, symbol: str) -> Optional[Dict]:
        """
        Cherche une opportunité de trading pour le symbole donné
        
        Args:
            symbol: Paire de trading
            
        Returns:
            Opportunité de trading ou None si aucune opportunité n'est trouvée
        """
        pass
    
    def log_trade(self, opportunity: Dict, order_result: Dict) -> None:
        """
        Enregistre les détails d'un trade dans un fichier JSON
        
        Args:
            opportunity: Opportunité de trading
            order_result: Résultat de l'ordre
        """
        # Créer un identifiant unique pour le trade
        trade_id = order_result.get("position_id", f"trade_{int(datetime.now().timestamp())}")
        
        # Préparer les données du trade
        trade_data = {
            "trade_id": trade_id,
            "symbol": opportunity.get("symbol"),
            "timestamp": datetime.now().isoformat(),
            "score": opportunity.get("score"),
            "reasoning": opportunity.get("reasoning"),
            "entry_price": order_result.get("entry_price"),
            "stop_loss": order_result.get("stop_loss_price"),
            "take_profit": order_result.get("take_profit_price"),
            "indicators": opportunity.get("indicators", {}),
            "market_conditions": opportunity.get("market_conditions", {})
        }
        
        # Enregistrer dans un fichier JSON
        filename = os.path.join(self.trades_dir, f"{trade_id}.json")
        
        try:
            with open(filename, 'w') as f:
                json.dump(trade_data, f, indent=2, default=str)
            
            logger.info(f"Trade enregistré: {filename}")
        except Exception as e:
            logger.error(f"Erreur lors de l'enregistrement du trade: {str(e)}")
    
    def update_trade_result(self, trade_id: str, result: Dict) -> None:
        """
        Met à jour le fichier de journal d'un trade avec les résultats
        
        Args:
            trade_id: ID du trade
            result: Résultat du trade
        """
        filename = os.path.join(self.trades_dir, f"{trade_id}.json")
        
        if not os.path.exists(filename):
            logger.warning(f"Fichier de trade non trouvé: {filename}")
            return
        
        try:
            # Charger les données existantes
            with open(filename, 'r') as f:
                trade_data = json.load(f)
            
            # Ajouter les résultats
            trade_data["close_timestamp"] = datetime.now().isoformat()
            trade_data["exit_price"] = result.get("exit_price")
            trade_data["pnl_percent"] = result.get("pnl_percent")
            trade_data["pnl_absolute"] = result.get("pnl_absolute")
            trade_data["trade_duration"] = result.get("trade_duration")
            trade_data["exit_reason"] = result.get("exit_reason")
            
            # Enregistrer les données mises à jour
            with open(filename, 'w') as f:
                json.dump(trade_data, f, indent=2, default=str)
            
            logger.info(f"Résultat du trade mis à jour: {filename}")
        except Exception as e:
            logger.error(f"Erreur lors de la mise à jour du résultat du trade: {str(e)}")
