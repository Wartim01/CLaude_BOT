"""
Agent de trading intelligent qui utilise les prédictions des modèles d'IA
pour prendre des décisions de trading optimales
"""
import os
import json
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Union, Any
from datetime import datetime, timedelta
import time
import traceback

from config.config import DATA_DIR, TRADING_CONFIG
from utils.logger import setup_logger
from core.position_manager import PositionManager
from core.order_manager import OrderManager
from core.exchange_client import ExchangeClient
from core.adaptive_risk_manager import AdaptiveRiskManager
from ai.prediction_orchestrator import PredictionOrchestrator

logger = setup_logger("trading_agent")

class TradingAgent:
    """
    Agent intelligent qui utilise les prédictions des modèles d'IA
    pour prendre des décisions de trading et gérer les positions
    """
    def __init__(self, 
               exchange_client: Optional[ExchangeClient] = None,
               position_manager: Optional[PositionManager] = None,
               order_manager: Optional[OrderManager] = None,
               prediction_orchestrator: Optional[PredictionOrchestrator] = None,
               risk_manager: Optional[AdaptiveRiskManager] = None):
        """
        Initialise l'agent de trading
        
        Args:
            exchange_client: Client d'échange pour les opérations de trading
            position_manager: Gestionnaire de positions
            order_manager: Gestionnaire d'ordres
            prediction_orchestrator: Orchestrateur de prédictions
            risk_manager: Gestionnaire de risque adaptatif
        """
        self.exchange_client = exchange_client
        self.position_manager = position_manager
        self.order_manager = order_manager
        self.prediction_orchestrator = prediction_orchestrator
        self.risk_manager = risk_manager
        
        # Configuration de trading par défaut
        self.config = {
            "default_symbol": "BTCUSDT",
            "default_timeframe": "15m",
            "confidence_threshold": 0.6,  # Seuil de confiance pour les trades
            "position_size_pct": 0.05,    # 5% du capital par défaut
            "max_open_positions": 3,      # Nombre maximum de positions ouvertes
            "use_stop_loss": True,        # Utiliser les stop-loss
            "stop_loss_pct": 2.0,         # Stop-loss par défaut (%)
            "use_take_profit": True,      # Utiliser les take-profit
            "take_profit_pct": 4.0,       # Take-profit par défaut (%)
            "auto_trade": False,          # Trading automatique activé
            "trading_enabled": True,      # Trading en général activé
            "dry_run": True,              # Mode simulation si True
            "risk_adjustment": True       # Ajustement du risque automatique
        }
        
        # Historique des trades
        self.trade_history = []
        
        # Statistiques de performance
        self.performance_stats = {
            "total_trades": 0,
            "winning_trades": 0,
            "losing_trades": 0,
            "win_rate": 0.0,
            "total_profit": 0.0,
            "total_loss": 0.0,
            "profit_factor": 0.0,
            "avg_profit": 0.0,
            "avg_loss": 0.0,
            "largest_profit": 0.0,
            "largest_loss": 0.0
        }
        
        # État du système
        self.state = {
            "is_trading": False,
            "last_prediction_time": None,
            "last_trade_time": None,
            "pending_orders": 0,
            "open_positions": 0,
            "auto_trade_enabled": self.config["auto_trade"],
            "dry_run": self.config["dry_run"],
            "errors": []
        }
        
        # Répertoires pour la sauvegarde des données
        self.data_dir = os.path.join(DATA_DIR, "trading_agent")
        self.config_path = os.path.join(self.data_dir, "trading_config.json")
        self.history_path = os.path.join(self.data_dir, "trade_history.json")
        self.stats_path = os.path.join(self.data_dir, "performance_stats.json")
        
        # Créer les répertoires si nécessaires
        os.makedirs(self.data_dir, exist_ok=True)
        
        # Charger la configuration et l'historique
        self._load_config()
        self._load_trade_history()
        self._load_performance_stats()
        
        logger.info("Agent de trading initialisé")
    
    def update_trading_pairs(self, pairs: List[str]) -> None:
        """
        Met à jour la liste des paires de trading suivies
        
        Args:
            pairs: Liste des paires à suivre
        """
        self.trading_pairs = pairs
        logger.info(f"Paires de trading mises à jour: {pairs}")
    
    def process_new_data(self, symbol: str, timeframe: str, data: pd.DataFrame) -> Dict:
        """
        Traite de nouvelles données et génère une décision de trading
        
        Args:
            symbol: Symbole de trading
            timeframe: Période de temps
            data: DataFrame avec les données OHLCV
            
        Returns:
            Résultat de la décision de trading
        """
        if not self.config["trading_enabled"]:
            return {
                "decision": "SKIP",
                "reason": "Trading désactivé"
            }
        
        if not self.prediction_orchestrator:
            return {
                "decision": "ERROR",
                "reason": "Orchestrateur de prédictions non disponible"
            }
        
        try:
            # 1. Obtenir une prédiction
            prediction = self.prediction_orchestrator.get_prediction(
                symbol=symbol,
                timeframe=timeframe,
                data=data
            )
            
            # Mettre à jour l'état
            self.state["last_prediction_time"] = datetime.now().isoformat()
            
            # 2. Analyser la prédiction pour prendre une décision
            decision = self._analyze_prediction(symbol, prediction)
            
            # 3. Si auto-trade est activé, exécuter automatiquement le trade
            if self.config["auto_trade"] and decision["action"] != "NONE":
                self._execute_trade_from_decision(decision)
            
            return decision
            
        except Exception as e:
            logger.error(f"Erreur lors du traitement des données pour {symbol}: {str(e)}")
            logger.error(traceback.format_exc())
            
            self.state["errors"].append({
                "time": datetime.now().isoformat(),
                "error": str(e),
                "context": f"process_new_data:{symbol}:{timeframe}"
            })
            
            return {
                "decision": "ERROR",
                "reason": f"Erreur: {str(e)}"
            }
    
    def _analyze_prediction(self, symbol: str, prediction: Dict) -> Dict:
        """
        Analyse une prédiction pour générer une décision de trading
        
        Args:
            symbol: Symbole de trading
            prediction: Prédiction du modèle
            
        Returns:
            Décision de trading
        """
        # Vérifier si la prédiction contient une erreur
        if "error" in prediction:
            return {
                "action": "NONE",
                "reason": f"Erreur dans la prédiction: {prediction['error']}",
                "confidence": 0.0,
                "prediction": prediction
            }
        
        # Extraire les informations principales
        primary = prediction.get("primary", {})
        direction = primary.get("direction", "NEUTRAL")
        confidence = primary.get("confidence", 0.0)
        signal_type = primary.get("signal_type", "NEUTRAL")
        
        # Vérifier le seuil de confiance
        confidence_threshold = self.config["confidence_threshold"]
        
        if confidence < confidence_threshold:
            return {
                "action": "NONE",
                "reason": f"Confiance insuffisante ({confidence:.2f} < {confidence_threshold})",
                "confidence": confidence,
                "symbol": symbol,
                "direction": direction,
                "prediction": prediction,
                "timestamp": datetime.now().isoformat()
            }
        
        # Vérifier les positions existantes
        if self.position_manager:
            open_positions = self.position_manager.get_open_positions(symbol)
            
            # Si position longue existe déjà et signal baissier fort
            if any(p["side"] == "BUY" for p in open_positions) and direction == "DOWN" and confidence > 0.7:
                return {
                    "action": "CLOSE_LONG",
                    "reason": f"Signal baissier fort ({confidence:.2f}) alors que position longue ouverte",
                    "confidence": confidence,
                    "symbol": symbol,
                    "direction": direction,
                    "signal_type": signal_type,
                    "prediction": prediction,
                    "timestamp": datetime.now().isoformat(),
                    "position_ids": [p["id"] for p in open_positions if p["side"] == "BUY"]
                }
            
            # Si position courte existe déjà et signal haussier fort
            if any(p["side"] == "SELL" for p in open_positions) and direction == "UP" and confidence > 0.7:
                return {
                    "action": "CLOSE_SHORT",
                    "reason": f"Signal haussier fort ({confidence:.2f}) alors que position courte ouverte",
                    "confidence": confidence,
                    "symbol": symbol,
                    "direction": direction,
                    "signal_type": signal_type,
                    "prediction": prediction,
                    "timestamp": datetime.now().isoformat(),
                    "position_ids": [p["id"] for p in open_positions if p["side"] == "SELL"]
                }
        
        # Vérifier le nombre maximum de positions
        max_positions = self.config["max_open_positions"]
        current_positions = 0
        
        if self.position_manager:
            current_positions = len(self.position_manager.get_open_positions())
        
        if current_positions >= max_positions:
            return {
                "action": "NONE",
                "reason": f"Nombre maximum de positions atteint ({current_positions}/{max_positions})",
                "confidence": confidence,
                "symbol": symbol,
                "direction": direction,
                "prediction": prediction,
                "timestamp": datetime.now().isoformat()
            }
        
        # Déterminer l'action en fonction de la direction
        if direction == "UP":
            return {
                "action": "BUY",
                "reason": f"Signal haussier avec confiance suffisante ({confidence:.2f})",
                "confidence": confidence,
                "symbol": symbol,
                "direction": direction,
                "signal_type": signal_type,
                "prediction": prediction,
                "timestamp": datetime.now().isoformat()
            }
        elif direction == "DOWN":
            return {
                "action": "SELL",
                "reason": f"Signal baissier avec confiance suffisante ({confidence:.2f})",
                "confidence": confidence,
                "symbol": symbol,
                "direction": direction,
                "signal_type": signal_type,
                "prediction": prediction,
                "timestamp": datetime.now().isoformat()
            }
        else:
            return {
                "action": "NONE",
                "reason": "Signal neutre",
                "confidence": confidence,
                "symbol": symbol,
                "direction": direction,
                "prediction": prediction,
                "timestamp": datetime.now().isoformat()
            }
    
    def _execute_trade_from_decision(self, decision: Dict) -> Dict:
        """
        Exécute un trade en fonction d'une décision
        
        Args:
            decision: Décision de trading
            
        Returns:
            Résultat de l'exécution
        """
        action = decision["action"]
        symbol = decision["symbol"]
        
        if action == "NONE":
            return {
                "success": True,
                "message": "Aucune action à exécuter"
            }
        
        # Fermer des positions existantes
        if action in ["CLOSE_LONG", "CLOSE_SHORT"]:
            if "position_ids" in decision and self.position_manager:
                position_ids = decision["position_ids"]
                
                results = []
                for position_id in position_ids:
                    result = self.position_manager.close_position(position_id)
                    results.append(result)
                
                # Enregistrer le trade
                self._record_trade({
                    "symbol": symbol,
                    "action": action,
                    "positions_closed": position_ids,
                    "results": results,
                    "timestamp": datetime.now().isoformat(),
                    "prediction": decision["prediction"],
                    "reason": decision["reason"]
                })
                
                return {
                    "success": True,
                    "message": f"Positions fermées: {position_ids}",
                    "results": results
                }
        
        # Ouvrir une nouvelle position
        if action in ["BUY", "SELL"] and self.position_manager:
            # Calculer la taille de position
            position_size = self._calculate_position_size(symbol, decision)
            
            # Calculer les niveaux de stop-loss et take-profit
            stop_loss, take_profit = self._calculate_exit_levels(symbol, action, decision)
            
            # Créer la position
            try:
                result = self.position_manager.create_position(
                    symbol=symbol,
                    side=action,
                    quantity=position_size,
                    stop_loss=stop_loss,
                    take_profit=take_profit
                )
                
                # Enregistrer le trade
                self._record_trade({
                    "symbol": symbol,
                    "action": action,
                    "position_id": result.get("position_id"),
                    "size": position_size,
                    "stop_loss": stop_loss,
                    "take_profit": take_profit,
                    "timestamp": datetime.now().isoformat(),
                    "prediction": decision["prediction"],
                    "reason": decision["reason"],
                    "confidence": decision["confidence"]
                })
                
                # Mettre à jour l'état
                self.state["last_trade_time"] = datetime.now().isoformat()
                self.state["open_positions"] = len(self.position_manager.get_open_positions())
                
                return {
                    "success": True,
                    "message": f"Position {action} créée pour {symbol}",
                    "result": result
                }
                
            except Exception as e:
                logger.error(f"Erreur lors de la création de la position {action} pour {symbol}: {str(e)}")
                
                self.state["errors"].append({
                    "time": datetime.now().isoformat(),
                    "error": str(e),
                    "context": f"execute_trade:{action}:{symbol}"
                })
                
                return {
                    "success": False,
                    "error": str(e)
                }
        
        return {
            "success": False,
            "error": f"Action non prise en charge: {action}"
        }
    
    def _calculate_position_size(self, symbol: str, decision: Dict) -> float:
        """
        Calcule la taille optimale de la position
        
        Args:
            symbol: Symbole de trading
            decision: Décision de trading
            
        Returns:
            Taille de la position
        """
        # Taille de position par défaut (pourcentage du capital)
        position_size_pct = self.config["position_size_pct"]
        
        # Si un gestionnaire de risque est disponible, l'utiliser
        if self.risk_manager:
            # La confiance de la prédiction peut influencer la taille
            confidence = decision.get("confidence", 0.5)
            
            # Vérifier si un ajustement du risque est nécessaire
            if self.config["risk_adjustment"]:
                position_size = self.risk_manager.calculate_position_size(
                    symbol=symbol,
                    confidence=confidence,
                    market_volatility=None  # On pourrait ajouter d'autres paramètres
                )
                
                return position_size
        
        # Calcul simple par défaut
        account_balance = 1000.0  # Valeur par défaut
        
        if self.exchange_client:
            try:
                account_info = self.exchange_client.get_account_info()
                account_balance = account_info.get("balance", 1000.0)
            except Exception as e:
                logger.warning(f"Impossible d'obtenir la balance du compte: {str(e)}")
        
        position_size = account_balance * position_size_pct
        
        return position_size
    
    def _calculate_exit_levels(self, symbol: str, action: str, decision: Dict) -> Tuple[float, float]:
        """
        Calcule les niveaux de stop-loss et take-profit
        
        Args:
            symbol: Symbole de trading
            action: Action de trading (BUY/SELL)
            decision: Décision de trading
            
        Returns:
            Tuple (stop_loss, take_profit)
        """
        # Obtenir le prix actuel
        current_price = 0.0
        
        if self.exchange_client:
            try:
                ticker = self.exchange_client.get_ticker(symbol)
                current_price = ticker.get("last", 0.0)
            except Exception as e:
                logger.warning(f"Impossible d'obtenir le prix actuel pour {symbol}: {str(e)}")
        
        if current_price == 0.0:
            raise ValueError(f"Prix actuel non disponible pour {symbol}")
        
        # Pourcentages par défaut
        stop_loss_pct = self.config["stop_loss_pct"] / 100.0
        take_profit_pct = self.config["take_profit_pct"] / 100.0
        
        # Ajuster les niveaux en fonction de la confiance
        confidence = decision.get("confidence", 0.5)
        
        # Plus la confiance est élevée, plus le stop-loss peut être serré
        # et le take-profit éloigné
        stop_loss_pct = stop_loss_pct * (1.1 - confidence * 0.2)
        take_profit_pct = take_profit_pct * (0.9 + confidence * 0.2)
        
        # Calculer les niveaux absolus
        if action == "BUY":
            stop_loss = current_price * (1 - stop_loss_pct)
            take_profit = current_price * (1 + take_profit_pct)
        else:  # SELL
            stop_loss = current_price * (1 + stop_loss_pct)
            take_profit = current_price * (1 - take_profit_pct)
        
        return stop_loss, take_profit
    
    def manual_trade(self, symbol: str, direction: str, quantity: Optional[float] = None,
                   reason: str = "Manual trade") -> Dict:
        """
        Exécute un trade manuellement
        
        Args:
            symbol: Symbole de trading
            direction: Direction du trade (BUY/SELL)
            quantity: Quantité à trader (calculée automatiquement si None)
            reason: Raison du trade
            
        Returns:
            Résultat de l'exécution
        """
        if not self.config["trading_enabled"]:
            return {
                "success": False,
                "error": "Trading désactivé"
            }
        
        if direction not in ["BUY", "SELL", "CLOSE"]:
            return {
                "success": False,
                "error": f"Direction invalide: {direction}"
            }
        
        try:
            if direction == "CLOSE":
                # Fermer toutes les positions sur ce symbole
                if not self.position_manager:
                    return {
                        "success": False,
                        "error": "Gestionnaire de positions non disponible"
                    }
                
                open_positions = self.position_manager.get_open_positions(symbol)
                
                if not open_positions:
                    return {
                        "success": False,
                        "error": f"Aucune position ouverte pour {symbol}"
                    }
                
                results = []
                for position in open_positions:
                    result = self.position_manager.close_position(position["id"])
                    results.append(result)
                
                # Enregistrer le trade
                self._record_trade({
                    "symbol": symbol,
                    "action": "CLOSE",
                    "positions_closed": [p["id"] for p in open_positions],
                    "results": results,
                    "timestamp": datetime.now().isoformat(),
                    "reason": reason
                })
                
                return {
                    "success": True,
                    "message": f"Positions fermées: {len(results)}",
                    "results": results
                }
            
            # Pour BUY ou SELL
            if not self.position_manager:
                return {
                    "success": False,
                    "error": "Gestionnaire de positions non disponible"
                }
            
            # Calculer la quantité si nécessaire
            if quantity is None:
                # Créer une décision fictive pour le calcul
                mock_decision = {
                    "action": direction,
                    "confidence": 0.75,
                    "symbol": symbol
                }
                
                quantity = self._calculate_position_size(symbol, mock_decision)
            
            # Calculer les niveaux de stop-loss et take-profit
            mock_decision = {
                "action": direction,
                "confidence": 0.75,
                "symbol": symbol
            }
            
            stop_loss, take_profit = self._calculate_exit_levels(symbol, direction, mock_decision)
            
            # Créer la position
            result = self.position_manager.create_position(
                symbol=symbol,
                side=direction,
                quantity=quantity,
                stop_loss=stop_loss,
                take_profit=take_profit
            )
            
            # Enregistrer le trade
            self._record_trade({
                "symbol": symbol,
                "action": direction,
                "position_id": result.get("position_id"),
                "size": quantity,
                "stop_loss": stop_loss,
                "take_profit": take_profit
            })
            
            return {
                "success": True,
                "message": f"Trade manuel {direction} exécuté pour {symbol}",
                "result": result
            }
            
        except Exception as e:
            logger.error(f"Erreur lors de l'exécution du trade manuel pour {symbol}: {str(e)}")
            
            self.state["errors"].append({
                "time": datetime.now().isoformat(),
                "error": str(e),
                "context": f"manual_trade:{direction}:{symbol}"
            })
            
            return {
                "success": False,
                "error": str(e)
            }
    
    def _record_trade(self, trade: Dict) -> None:
        """
        Enregistre un trade dans l'historique
        
        Args:
            trade: Dictionnaire contenant les informations du trade
        """
        self.trade_history.append(trade)
        
        # Mettre à jour les statistiques de performance
        self._update_performance_stats(trade)
        
        # Sauvegarder l'historique
        self._save_trade_history()
    
    def _update_performance_stats(self, trade: Dict) -> None:
        """
        Met à jour les statistiques de performance en fonction d'un trade
        
        Args:
            trade: Dictionnaire contenant les informations du trade
        """
        self.performance_stats["total_trades"] += 1
        
        if "results" in trade:
            for result in trade["results"]:
                if result.get("success", False):
                    self.performance_stats["winning_trades"] += 1
                    self.performance_stats["total_profit"] += result.get("profit", 0.0)
                else:
                    self.performance_stats["losing_trades"] += 1
                    self.performance_stats["total_loss"] += result.get("loss", 0.0)
        
        # Calculer le taux de réussite
        total_trades = self.performance_stats["total_trades"]
        winning_trades = self.performance_stats["winning_trades"]
        
        if total_trades > 0:
            self.performance_stats["win_rate"] = (winning_trades / total_trades) * 100.0
        
        # Calculer le facteur de profit
        total_profit = self.performance_stats["total_profit"]
        total_loss = self.performance_stats["total_loss"]
        
        if total_loss != 0:
            self.performance_stats["profit_factor"] = total_profit / abs(total_loss)
        
        # Calculer les profits et pertes moyens
        if winning_trades > 0:
            self.performance_stats["avg_profit"] = total_profit / winning_trades
        
        if self.performance_stats["losing_trades"] > 0:
            self.performance_stats["avg_loss"] = total_loss / self.performance_stats["losing_trades"]
        
        # Mettre à jour les plus grands profits et pertes
        if "results" in trade:
            for result in trade["results"]:
                profit = result.get("profit", 0.0)
                loss = result.get("loss", 0.0)
                
                if profit > self.performance_stats["largest_profit"]:
                    self.performance_stats["largest_profit"] = profit
                
                if loss < self.performance_stats["largest_loss"]:
                    self.performance_stats["largest_loss"] = loss
        
        # Sauvegarder les statistiques de performance
        self._save_performance_stats()
    
    def _load_config(self) -> None:
        """
        Charge la configuration de trading depuis un fichier
        """
        if os.path.exists(self.config_path):
            with open(self.config_path, 'r') as f:
                self.config = json.load(f)
            logger.info("Configuration de trading chargée")
    
    def _save_config(self) -> None:
        """
        Sauvegarde la configuration de trading dans un fichier
        """
        with open(self.config_path, 'w') as f:
            json.dump(self.config, f, indent=2)
        logger.info("Configuration de trading sauvegardée")
    
    def _load_trade_history(self) -> None:
        """
        Charge l'historique des trades depuis un fichier
        """
        if os.path.exists(self.history_path):
            with open(self.history_path, 'r') as f:
                self.trade_history = json.load(f)
            logger.info("Historique des trades chargé")
    
    def _save_trade_history(self) -> None:
        """
        Sauvegarde l'historique des trades dans un fichier
        """
        with open(self.history_path, 'w') as f:
            json.dump(self.trade_history, f, indent=2)
        logger.info("Historique des trades sauvegardé")
    
    def _load_performance_stats(self) -> None:
        """
        Charge les statistiques de performance depuis un fichier
        """
        if os.path.exists(self.stats_path):
            with open(self.stats_path, 'r') as f:
                self.performance_stats = json.load(f)
            logger.info("Statistiques de performance chargées")
    
    def _save_performance_stats(self) -> None:
        """
        Sauvegarde les statistiques de performance dans un fichier
        """
        with open(self.stats_path, 'w') as f:
            json.dump(self.performance_stats, f, indent=2)
        logger.info("Statistiques de performance sauvegardées")