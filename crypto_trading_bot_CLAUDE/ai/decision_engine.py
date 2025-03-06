"""
Moteur de décision qui combine les prédictions de l'IA, l'analyse technique,
et les indicateurs de risque du marché pour générer des décisions de trading optimales
"""
import os
import json
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Union, Any
from datetime import datetime, timedelta

from config.config import DATA_DIR
from utils.logger import setup_logger
from ai.trading_agent import TradingAgent
from ai.strategy_integrator import StrategyIntegrator
from utils.market_risk_feed import MarketRiskFeed

logger = setup_logger("decision_engine")

class DecisionEngine:
    """
    Moteur de décision qui analyse toutes les sources d'information
    pour générer les meilleures décisions de trading possibles
    """
    def __init__(self,
                trading_agent: Optional[TradingAgent] = None,
                strategy_integrator: Optional[StrategyIntegrator] = None,
                market_risk_feed: Optional[MarketRiskFeed] = None):
        """
        Initialise le moteur de décision
        
        Args:
            trading_agent: Agent de trading
            strategy_integrator: Intégrateur de stratégies
            market_risk_feed: Flux d'information sur le risque de marché
        """
        self.trading_agent = trading_agent
        self.strategy_integrator = strategy_integrator
        self.market_risk_feed = market_risk_feed
        
        # Historique des décisions
        self.decision_history = {}
        
        # Configuration des poids
        self.decision_weights = {
            "strategy_signal": 0.50,  # Signal de stratégie intégrée
            "market_risk": 0.30,      # Risque de marché
            "technical_score": 0.20    # Score technique
        }
        
        # Seuils de décision
        self.decision_thresholds = {
            "strong_buy": 0.7,
            "buy": 0.55,
            "neutral": 0.45,
            "sell": 0.55,
            "strong_sell": 0.7
        }
        
        # Répertoire de données
        self.data_dir = os.path.join(DATA_DIR, "decisions")
        self.history_path = os.path.join(self.data_dir, "decision_history.json")
        self.performance_path = os.path.join(self.data_dir, "decision_performance.json")
        
        # Créer les répertoires si nécessaires
        os.makedirs(self.data_dir, exist_ok=True)
        
        # Chargement de l'historique
        self._load_decision_history()
        
        # Dernier risque de marché connu
        self.last_known_market_risk = {
            "level": "medium",
            "score": 50.0,
            "timestamp": datetime.now().isoformat()
        }
        
        # Performance globale
        self.performance_metrics = {
            "total_decisions": 0,
            "correct_decisions": 0,
            "incorrect_decisions": 0,
            "total_pnl": 0.0,
            "win_rate": 0.0
        }
    
    def evaluate_trading_opportunity(self, symbol: str, data: pd.DataFrame, 
                                 timeframe: str = '1h', execute: bool = False) -> Dict:
        """
        Évalue une opportunité de trading en combinant toutes les sources d'information
        
        Args:
            symbol: Symbole de trading (ex: 'BTCUSDT')
            data: DataFrame avec les données OHLCV
            timeframe: Timeframe des données
            execute: Exécuter automatiquement le trade si opportunité valide
            
        Returns:
            Résultat de l'évaluation et décision de trading
        """
        current_time = datetime.now()
        
        # 1. Récupérer le signal de la stratégie intégrée
        strategy_signal = self._get_strategy_signal(symbol, data, timeframe)
        
        # 2. Récupérer les indicateurs de risque du marché
        market_risk = self._get_market_risk(symbol)
        
        # 3. Obtenir le score technique direct
        technical_score = self._get_technical_score(data)
        
        # 4. Combiner toutes les sources pour une décision finale
        decision = self._make_decision(
            symbol=symbol,
            strategy_signal=strategy_signal,
            market_risk=market_risk,
            technical_score=technical_score
        )
        
        # 5. Enregistrer la décision dans l'historique
        decision_id = self._record_decision(
            symbol=symbol,
            timeframe=timeframe,
            decision=decision,
            strategy_signal=strategy_signal,
            market_risk=market_risk,
            technical_score=technical_score
        )
        
        # 6. Exécuter le trade si demandé et si la décision est favorable
        execution_result = None
        if execute and decision["should_trade"] and self.trading_agent:
            execution_result = self._execute_trade(
                symbol=symbol,
                decision=decision
            )
            
            decision["execution_result"] = execution_result
        
        return decision
    
    def _get_strategy_signal(self, symbol: str, data: pd.DataFrame, timeframe: str) -> Dict:
        """
        Récupère le signal de stratégie intégrée
        
        Args:
            symbol: Symbole de trading
            data: DataFrame avec les données OHLCV
            timeframe: Timeframe des données
            
        Returns:
            Signal de stratégie
        """
        if not self.strategy_integrator:
            return {
                "available": False,
                "reason": "Strategy integrator not available"
            }
        
        try:
            # Récupérer le signal de l'intégrateur de stratégies
            signal = self.strategy_integrator.generate_trade_signal(
                symbol=symbol,
                data=data,
                timeframe=timeframe
            )
            
            return {
                "available": True,
                "direction": signal.get("direction", "NEUTRAL"),
                "signal_type": signal.get("signal_type", "NEUTRAL"),
                "confidence": signal.get("confidence", 0.0),
                "strength": signal.get("strength", 0),
                "signals": signal.get("signals", []),
                "raw_signal": signal
            }
        
        except Exception as e:
            logger.error(f"Erreur lors de la récupération du signal de stratégie pour {symbol}: {str(e)}")
            
            return {
                "available": False,
                "reason": f"Error: {str(e)}"
            }
    
    def _get_market_risk(self, symbol: str) -> Dict:
        """
        Récupère les indicateurs de risque du marché
        
        Args:
            symbol: Symbole de trading
            
        Returns:
            Indicateurs de risque du marché
        """
        if not self.market_risk_feed:
            return {
                "available": False,
                "reason": "Market risk feed not available",
                "fallback": self.last_known_market_risk
            }
        
        try:
            # Récupérer le risque du marché
            risk_data = self.market_risk_feed.get_market_risk(symbol)
            
            # Mettre à jour le dernier risque connu
            if risk_data.get("level") is not None:
                self.last_known_market_risk = {
                    "level": risk_data["level"],
                    "score": risk_data["risk_score"],
                    "timestamp": datetime.now().isoformat()
                }
            
            return {
                "available": True,
                "level": risk_data.get("level", "medium"),
                "risk_score": risk_data.get("risk_score", 50.0),
                "risk_factors": risk_data.get("risk_factors", []),
                "market_conditions": risk_data.get("market_conditions", {}),
                "raw_data": risk_data
            }
        
        except Exception as e:
            logger.error(f"Erreur lors de la récupération du risque de marché pour {symbol}: {str(e)}")
            
            return {
                "available": False,
                "reason": f"Error: {str(e)}",
                "fallback": self.last_known_market_risk
            }
    
    def _get_technical_score(self, data: pd.DataFrame) -> Dict:
        """
        Calcule un score technique direct basé sur les données
        
        Args:
            data: DataFrame avec les données OHLCV
            
        Returns:
            Score technique
        """
        if data is None or data.empty:
            return {
                "available": False,
                "reason": "No data provided"
            }
        
        try:
            # Import ici pour éviter les imports circulaires
            from strategies.market_state import analyze_market_state
            from strategies.technical_bounce import detect_technical_bounce
            
            # Analyser l'état du marché
            market_state = analyze_market_state(data)
            
            # Détecter les rebounds techniques
            bounce_result = detect_technical_bounce(data, market_state)
            
            # Calculer un score simple basé sur ces analyses
            score = 50.0  # Neutre par défaut
            direction = "NEUTRAL"
            
            # Ajuster le score en fonction de l'état du marché
            if market_state.get("state") == "bullish":
                score += 15.0
            elif market_state.get("state") == "bearish":
                score -= 15.0
            
            # Ajuster le score en fonction des signaux de rebond
            bounce_signals = bounce_result.get("signals", [])
            if "bullish_bounce" in bounce_signals or "bullish_breakout" in bounce_signals:
                score += 10.0
            if "bearish_bounce" in bounce_signals or "bearish_breakdown" in bounce_signals:
                score -= 10.0
            
            # Déterminer la direction
            if score >= 70:
                direction = "BUY"
            elif score <= 30:
                direction = "SELL"
            
            return {
                "available": True,
                "score": score,
                "direction": direction,
                "market_state": market_state.get("state", "unknown"),
                "signals": bounce_signals
            }
        
        except Exception as e:
            logger.error(f"Erreur lors du calcul du score technique: {str(e)}")
            
            return {
                "available": False,
                "reason": f"Error: {str(e)}"
            }
    
    def _make_decision(self, symbol: str, 
                     strategy_signal: Dict,
                     market_risk: Dict,
                     technical_score: Dict) -> Dict:
        """
        Combine toutes les sources d'information pour prendre une décision
        
        Args:
            symbol: Symbole de trading
            strategy_signal: Signal de stratégie
            market_risk: Risque du marché
            technical_score: Score technique
            
        Returns:
            Décision de trading
        """
        # 1. Vérifier la disponibilité des données
        strategy_available = strategy_signal.get("available", False)
        risk_available = market_risk.get("available", False)
        technical_available = technical_score.get("available", False)
        
        # 2. Extraire et normaliser les signaux (0-100 scale où 50 est neutre)
        strategy_score = 50.0
        risk_score = 50.0
        tech_score = 50.0
        
        # 2.1 Score de stratégie
        if strategy_available:
            direction = strategy_signal.get("direction", "NEUTRAL")
            confidence = strategy_signal.get("confidence", 0.0)
            
            if direction == "BUY":
                strategy_score = 50 + (confidence * 50)
            elif direction == "SELL":
                strategy_score = 50 - (confidence * 50)
        
        # 2.2 Score de risque
        if risk_available:
            risk_score = market_risk.get("risk_score", 50.0)
            # Inverser le score de risque car un risque élevé signifie une position prudente
            risk_score = 100 - risk_score
        elif "fallback" in market_risk:
            risk_score = 100 - market_risk["fallback"].get("score", 50.0)
        
        # 2.3 Score technique
        if technical_available:
            tech_score = technical_score.get("score", 50.0)
        
        # 3. Appliquer les poids pour obtenir un score combiné
        strategy_weight = self.decision_weights.get("strategy_signal", 0.5)
        risk_weight = self.decision_weights.get("market_risk", 0.3)
        technical_weight = self.decision_weights.get("technical_score", 0.2)
        
        # Recalculer les poids si certaines données ne sont pas disponibles
        total_weight = 0.0
        
        if strategy_available:
            total_weight += strategy_weight
        else:
            strategy_weight = 0.0
            
        if risk_available or "fallback" in market_risk:
            total_weight += risk_weight
        else:
            risk_weight = 0.0
            
        if technical_available:
            total_weight += technical_weight
        else:
            technical_weight = 0.0
        
        # Éviter la division par zéro
        if total_weight == 0.0:
            return {
                "should_trade": False,
                "reason": "Données insuffisantes pour la prise de décision",
                "direction": "NEUTRAL",
                "confidence": 0.0,
                "weighted_score": 50.0
            }
        
        # Normaliser les poids restants
        strategy_weight /= total_weight
        risk_weight /= total_weight
        technical_weight /= total_weight
        
        # Calculer le score pondéré
        weighted_score = (
            (strategy_score * strategy_weight) +
            (risk_score * risk_weight) +
            (tech_score * technical_weight)
        )
        
        # 4. Interpréter le score pour obtenir une décision
        direction = "NEUTRAL"
        signal_type = "NEUTRAL"
        should_trade = False
        confidence = abs(weighted_score - 50) / 50.0  # 0 à 1 où 1 est la confiance maximale
        reason = "Score neutre"
        
        # Décision d'achat
        if weighted_score > 50:
            if confidence >= self.decision_thresholds["strong_buy"]:
                direction = "BUY"
                signal_type = "STRONG_BUY"
                should_trade = True
                reason = "Signal d'achat fort"
            elif confidence >= self.decision_thresholds["buy"]:
                direction = "BUY"
                signal_type = "BUY"
                should_trade = True
                reason = "Signal d'achat"
        # Décision de vente
        elif weighted_score < 50:
            if confidence >= self.decision_thresholds["strong_sell"]:
                direction = "SELL"
                signal_type = "STRONG_SELL"
                should_trade = True
                reason = "Signal de vente fort"
            elif confidence >= self.decision_thresholds["sell"]:
                direction = "SELL"
                signal_type = "SELL"
                should_trade = True
                reason = "Signal de vente"
        
        # 5. Vérifier si le risque de marché n'est pas trop élevé
        if risk_available and market_risk.get("level") == "extreme" and should_trade:
            should_trade = False
            reason = "Risque de marché extrême, trading suspendu"
        
        # 6. Créer la décision finale
        decision = {
            "should_trade": should_trade,
            "direction": direction,
            "signal_type": signal_type,
            "confidence": confidence,
            "weighted_score": weighted_score,
            "reason": reason,
            "timestamp": datetime.now().isoformat(),
            "contributors": {
                "strategy": {
                    "available": strategy_available,
                    "score": strategy_score,
                    "weight": strategy_weight * 100
                },
                "risk": {
                    "available": risk_available,
                    "score": risk_score,
                    "weight": risk_weight * 100
                },
                "technical": {
                    "available": technical_available,
                    "score": tech_score,
                    "weight": technical_weight * 100
                }
            }
        }
        
        return decision
    
    def _record_decision(self, symbol: str, timeframe: str, 
                       decision: Dict, strategy_signal: Dict,
                       market_risk: Dict, technical_score: Dict) -> str:
        """
        Enregistre une décision de trading pour analyse ultérieure
        
        Args:
            symbol: Symbole de trading
            timeframe: Timeframe des données
            decision: Décision finale
            strategy_signal: Signal de stratégie
            market_risk: Risque du marché
            technical_score: Score technique
            
        Returns:
            ID de la décision (timestamp)
        """
        # Créer un ID unique pour cette décision
        decision_id = f"{datetime.now().isoformat()}_{symbol}"
        
        # Créer l'entrée d'historique
        decision_entry = {
            "id": decision_id,
            "symbol": symbol,
            "timeframe": timeframe,
            "timestamp": datetime.now().isoformat(),
            "decision": decision,
            "inputs": {
                "strategy": strategy_signal,
                "risk": market_risk,
                "technical": technical_score
            },
            "result": {
                "actual_outcome": None,  # À remplir plus tard
                "pnl": None,             # À remplir plus tard
                "success": None          # À remplir plus tard
            }
        }
        
        # Ajouter à l'historique
        if symbol not in self.decision_history:
            self.decision_history[symbol] = []
            
        self.decision_history[symbol].append(decision_entry)
        
        # Limiter la taille de l'historique pour chaque symbole
        max_history = 100
        if len(self.decision_history[symbol]) > max_history:
            self.decision_history[symbol] = self.decision_history[symbol][-max_history:]
        
        # Sauvegarder l'historique
        self._save_decision_history()
        
        return decision_id
    
    def _execute_trade(self, symbol: str, decision: Dict) -> Dict:
        """
        Exécute un trade basé sur la décision
        
        Args:
            symbol: Symbole de trading
            decision: Décision de trading
            
        Returns:
            Résultat de l'exécution
        """
        if not self.trading_agent:
            return {
                "success": False,
                "error": "Agent de trading non disponible"
            }
        
        if not decision["should_trade"]:
            return {
                "success": False,
                "error": "La décision ne recommande pas de trader"
            }
        
        try:
            # Exécuter le trade via l'agent de trading
            result = self.trading_agent.manual_trade(
                symbol=symbol,
                direction=decision["direction"]
            )
            
            # Si l'exécution a réussi, enregistrer des métadonnées supplémentaires
            if result.get("success", False) and result.get("position_id"):
                # On pourrait ajouter des métadonnées supplémentaires ici
                # Par exemple, lier la décision au trade exécuté
                position_id = result.get("position_id")
                
                # Mise à jour de l'historique des décisions pour lier à la position
                for decisions in self.decision_history.values():
                    for d in decisions:
                        if d["id"] == decision.get("id"):
                            d["position_id"] = position_id
                            d["execution_time"] = datetime.now().isoformat()
                            break
                
                self._save_decision_history()
            
            return result
        
        except Exception as e:
            logger.error(f"Erreur lors de l'exécution du trade pour {symbol}: {str(e)}")
            
            return {
                "success": False,
                "error": f"Erreur d'exécution du trade: {str(e)}"
            }
    
    def update_decision_outcome(self, decision_id: str, actual_outcome: str, pnl: float) -> bool:
        """
        Met à jour l'historique avec le résultat réel d'une décision de trading
        
        Args:
            decision_id: ID de la décision
            actual_outcome: Résultat réel ("BUY", "SELL", "NEUTRAL")
            pnl: Profit/perte réalisé
            
        Returns:
            Succès de la mise à jour
        """
        # Rechercher la décision dans l'historique
        found_decision = False
        
        for symbol, decisions in self.decision_history.items():
            for decision in decisions:
                if decision["id"] == decision_id:
                    # Mettre à jour le résultat
                    decision["result"]["actual_outcome"] = actual_outcome
                    decision["result"]["pnl"] = pnl
                    
                    # Déterminer si la décision était correcte
                    predicted_direction = decision["decision"]["direction"]
                    success = (predicted_direction == actual_outcome and predicted_direction != "NEUTRAL")
                    decision["result"]["success"] = success
                    
                    # Mettre à jour les métriques de performance
                    self.performance_metrics["total_decisions"] += 1
                    
                    if success:
                        self.performance_metrics["correct_decisions"] += 1
                    else:
                        self.performance_metrics["incorrect_decisions"] += 1
                    
                    self.performance_metrics["total_pnl"] += pnl
                    
                    if self.performance_metrics["total_decisions"] > 0:
                        self.performance_metrics["win_rate"] = (
                            self.performance_metrics["correct_decisions"] / 
                            self.performance_metrics["total_decisions"] * 100
                        )
                    
                    # Mettre à jour la performance dans les composants sous-jacents
                    self._update_component_performance(decision, actual_outcome, pnl)
                    
                    # Sauvegarder les données
                    self._save_decision_history()
                    self._save_performance_metrics()
                    
                    found_decision = True
                    break
            
            if found_decision:
                break
        
        if not found_decision:
            logger.warning(f"Décision non trouvée pour la mise à jour: {decision_id}")
            
        return found_decision
    
    def _update_component_performance(self, decision: Dict, actual_outcome: str, pnl: float) -> None:
        """
        Met à jour les performances des composants sous-jacents
        
        Args:
            decision: Décision complète
            actual_outcome: Résultat réel
            pnl: Profit/perte réalisé
        """
        # 1. Mettre à jour l'intégrateur de stratégies
        if (self.strategy_integrator and 
            "strategy" in decision["inputs"] and 
            decision["inputs"]["strategy"].get("available", False) and
            "raw_signal" in decision["inputs"]["strategy"]):
            
            try:
                strategy_signal = decision["inputs"]["strategy"]["raw_signal"]
                signal_timestamp = strategy_signal.get("timestamp")
                
                if signal_timestamp:
                    self.strategy_integrator.update_performance(
                        symbol=decision["symbol"],
                        signal_timestamp=signal_timestamp,
                        actual_outcome=actual_outcome,
                        pnl=pnl
                    )
            except Exception as e:
                logger.error(f"Erreur lors de la mise à jour des performances de l'intégrateur de stratégies: {str(e)}")
        
        # 2. Mettre à jour l'analyseur de sentiment des nouvelles
        # Cette partie serait à implémenter si l'analyseur de sentiment suivait ses performances
        
        # 3. Mettre à jour le flux de risque du marché
        # Cette partie serait à implémenter si le flux de risque ajustait ses modèles
    
    def get_performance(self) -> Dict:
        """
        Récupère les métriques de performance du moteur de décision
        
        Returns:
            Métriques de performance
        """
        # Calculer des métriques additionnelles
        avg_pnl_per_trade = 0
        if self.performance_metrics["total_decisions"] > 0:
            avg_pnl_per_trade = self.performance_metrics["total_pnl"] / self.performance_metrics["total_decisions"]
        
        # Préparer les graphiques de performance
        performance_by_symbol = {}
        for symbol, decisions in self.decision_history.items():
            completed_decisions = [d for d in decisions if d["result"]["pnl"] is not None]
            
            if completed_decisions:
                total_trades = len(completed_decisions)
                wins = len([d for d in completed_decisions if d["result"]["success"]])
                pnl_sum = sum(d["result"]["pnl"] for d in completed_decisions)
                
                performance_by_symbol[symbol] = {
                    "total_trades": total_trades,
                    "wins": wins,
                    "win_rate": (wins / total_trades * 100) if total_trades > 0 else 0,
                    "total_pnl": pnl_sum,
                    "avg_pnl": pnl_sum / total_trades if total_trades > 0 else 0
                }
        
        # Regrouper toutes les métriques
        performance = {
            "overall": {
                "total_decisions": self.performance_metrics["total_decisions"],
                "correct_decisions": self.performance_metrics["correct_decisions"],
                "incorrect_decisions": self.performance_metrics["incorrect_decisions"],
                "win_rate": self.performance_metrics["win_rate"],
                "total_pnl": self.performance_metrics["total_pnl"],
                "avg_pnl_per_trade": avg_pnl_per_trade
            },
            "by_symbol": performance_by_symbol,
            "components": {
                "strategy_integrator": (
                    self.strategy_integrator.get_performance_metrics()
                    if self.strategy_integrator else None
                ),
                "trading_agent": (
                    self.trading_agent.get_status()
                    if self.trading_agent else None
                )
            },
            "last_update": datetime.now().isoformat()
        }
        
        return performance
    
    def _load_decision_history(self) -> None:
        """Charge l'historique des décisions depuis le disque"""
        if os.path.exists(self.history_path):
            try:
                with open(self.history_path, 'r') as f:
                    self.decision_history = json.load(f)
                logger.info(f"Historique des décisions chargé: {sum(len(decisions) for decisions in self.decision_history.values())} décisions")
                
                # Vérifier le format après chargement
                for symbol, decisions in list(self.decision_history.items()):
                    if not isinstance(decisions, list):
                        logger.warning(f"Format incorrect pour le symbole {symbol}, réinitialisation")
                        self.decision_history[symbol] = []
            except Exception as e:
                logger.error(f"Erreur lors du chargement de l'historique des décisions: {str(e)}")
                self.decision_history = {}
        else:
            logger.info("Aucun historique de décisions trouvé, création d'un nouveau")
            self.decision_history = {}
        
        # Charger également les métriques de performance
        if os.path.exists(self.performance_path):
            try:
                with open(self.performance_path, 'r') as f:
                    self.performance_metrics = json.load(f)
            except Exception as e:
                logger.error(f"Erreur lors du chargement des métriques de performance: {str(e)}")
    
    def _save_decision_history(self) -> None:
        """Sauvegarde l'historique des décisions sur le disque"""
        try:
            with open(self.history_path, 'w') as f:
                json.dump(self.decision_history, f, indent=2)
        except Exception as e:
            logger.error(f"Erreur lors de la sauvegarde de l'historique des décisions: {str(e)}")
    
    def _save_performance_metrics(self) -> None:
        """Sauvegarde les métriques de performance sur le disque"""
        try:
            with open(self.performance_path, 'w') as f:
                json.dump(self.performance_metrics, f, indent=2)
        except Exception as e:
            logger.error(f"Erreur lors de la sauvegarde des métriques de performance: {str(e)}")
    
    def get_recent_decisions(self, symbol: Optional[str] = None, limit: int = 10) -> List[Dict]:
        """
        Récupère les décisions récentes pour un symbole ou tous les symboles
        
        Args:
            symbol: Symbole spécifique (optionnel)
            limit: Nombre maximum de décisions à récupérer
            
        Returns:
            Liste des décisions récentes
        """
        recent_decisions = []
        
        if symbol:
            # Récupérer les décisions pour un symbole spécifique
            if symbol in self.decision_history:
                recent_decisions = self.decision_history[symbol][-limit:]
        else:
            # Récupérer les décisions pour tous les symboles
            all_decisions = []
            for sym, decisions in self.decision_history.items():
                all_decisions.extend(decisions)
            
            # Trier par timestamp (du plus récent au plus ancien)
            all_decisions.sort(key=lambda x: x.get("timestamp", ""), reverse=True)
            recent_decisions = all_decisions[:limit]
        
        return recent_decisions
    
    def analyze_decision_patterns(self) -> Dict:
        """
        Analyse les patterns dans les décisions de trading pour identifier des tendances
        
        Returns:
            Analyse des patterns de décision
        """
        # Cette fonction pourrait être développée davantage pour fournir des insights
        # plus profonds sur les patterns de décision et leurs performances
        
        if not self.decision_history:
            return {"error": "Pas d'historique de décisions disponible pour l'analyse"}
        
        # Compter les décisions par direction et par résultat
        direction_counts = {"BUY": 0, "SELL": 0, "NEUTRAL": 0}
        success_by_direction = {"BUY": 0, "SELL": 0, "NEUTRAL": 0}
        pnl_by_direction = {"BUY": 0.0, "SELL": 0.0, "NEUTRAL": 0.0}
        
        # Analyser toutes les décisions complétées (avec un résultat)
        completed_decisions = []
        for symbol, decisions in self.decision_history.items():
            for decision in decisions:
                if decision["result"]["actual_outcome"] is not None:
                    completed_decisions.append(decision)
                    
                    direction = decision["decision"]["direction"]
                    success = decision["result"]["success"]
                    pnl = decision["result"]["pnl"] or 0.0
                    
                    direction_counts[direction] += 1
                    
                    if success:
                        success_by_direction[direction] += 1
                    
                    pnl_by_direction[direction] += pnl
        
        # Calculer les taux de réussite et les PnL moyens
        win_rates = {}
        avg_pnl = {}
        
        for direction in ["BUY", "SELL", "NEUTRAL"]:
            if direction_counts[direction] > 0:
                win_rates[direction] = (
                    success_by_direction[direction] / direction_counts[direction] * 100
                )
                avg_pnl[direction] = pnl_by_direction[direction] / direction_counts[direction]
            else:
                win_rates[direction] = 0.0
                avg_pnl[direction] = 0.0
        
        # Identifier les conditions de marché qui ont le mieux fonctionné
        # Cela nécessite d'analyser les entrées de décision (market_risk, sentiment, etc.)
        best_conditions = {"state": "unknown", "confidence": 0.0}
        
        # Cette analyse pourrait être plus sophistiquée
        # Pour l'instant, on identifie simplement l'état du marché qui a donné le meilleur win rate
        market_states = {}
        
        for decision in completed_decisions:
            if "technical" in decision["inputs"] and decision["inputs"]["technical"].get("available", False):
                market_state = decision["inputs"]["technical"].get("market_state", "unknown")
                
                if market_state not in market_states:
                    market_states[market_state] = {"count": 0, "success": 0}
                
                market_states[market_state]["count"] += 1
                
                if decision["result"]["success"]:
                    market_states[market_state]["success"] += 1
        
        # Trouver l'état de marché avec le meilleur win rate
        best_win_rate = 0.0
        for state, data in market_states.items():
            if data["count"] > 5:  # Minimum 5 décisions pour être significatif
                win_rate = data["success"] / data["count"] * 100
                if win_rate > best_win_rate:
                    best_win_rate = win_rate
                    best_conditions["state"] = state
                    best_conditions["confidence"] = win_rate / 100
        
        return {
            "total_completed_decisions": len(completed_decisions),
            "direction_counts": direction_counts,
            "win_rates": win_rates,
            "avg_pnl": avg_pnl,
            "best_conditions": best_conditions,
            "timestamp": datetime.now().isoformat()
        }
    
    def adjust_weights(self, new_weights: Dict = None) -> Dict:
        """
        Ajuste les poids des différentes sources d'information
        
        Args:
            new_weights: Nouveaux poids à appliquer
            
        Returns:
            Poids actuels après ajustement
        """
        if new_weights:
            # Vérifier que les poids sont valides
            required_keys = ["strategy_signal", "market_risk", "technical_score"]
            
            if not all(key in new_weights for key in required_keys):
                return {
                    "success": False,
                    "error": "Tous les poids requis ne sont pas présents",
                    "required_keys": required_keys,
                    "current_weights": self.decision_weights
                }
            
            # Vérifier que la somme est proche de 1.0
            weight_sum = sum(new_weights.values())
            if abs(weight_sum - 1.0) > 0.01:
                # Normaliser les poids pour qu'ils somment à 1.0
                for key in new_weights:
                    new_weights[key] /= weight_sum
            
            # Appliquer les nouveaux poids
            self.decision_weights = new_weights
            
            logger.info(f"Poids des décisions mis à jour: {self.decision_weights}")
        
        return {
            "success": True,
            "weights": self.decision_weights
        }