"""
Intégrateur de stratégies qui combine les signaux de différentes stratégies de trading
pour générer des décisions plus robustes basées sur des règles et des modèles d'IA
"""
import os
import json
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Union, Any
from datetime import datetime, timedelta
import importlib
import inspect

from config.config import DATA_DIR, STRATEGIES_CONFIG
from utils.logger import setup_logger

logger = setup_logger("strategy_integrator")

class StrategyIntegrator:
    """
    Intègre des signaux de stratégies diverses pour une meilleure prise de décision
    """
    def __init__(self, config_path: Optional[str] = None, strategies_dir: Optional[str] = None):
        """
        Initialise l'intégrateur de stratégies
        
        Args:
            config_path: Chemin du fichier de configuration
            strategies_dir: Répertoire contenant les modules de stratégie
        """
        # Chemins des fichiers
        self.config_path = config_path or os.path.join(DATA_DIR, "strategies", "strategy_config.json")
        self.strategies_dir = strategies_dir or os.path.join(os.path.dirname(os.path.dirname(__file__)), "strategies")
        self.performance_path = os.path.join(DATA_DIR, "strategies", "strategy_performance.json")
        
        # Assurer que les répertoires existent
        os.makedirs(os.path.dirname(self.config_path), exist_ok=True)
        
        # Listes des modules de stratégie
        self.strategies = {}
        self.active_strategies = []
        
        # Configuration
        self.config = {
            "default_weights": {},
            "adaptive_weights": True,
            "performance_window": 100,  # Nombre de trades pour l'évaluation
            "min_samples": 10,          # Nombre minimum de trades pour l'adaptation
            "confidence_threshold": 0.5, # Seuil minimum de confiance
            "majority_threshold": 0.6,  # Proportion pour majorité qualifiée
            "consensus_boost": 0.2,     # Bonus de confiance si consensus
            "disagreement_penalty": 0.1  # Pénalité si désaccord important
        }
        
        # Performance des stratégies
        self.performance = {}
        
        # Historique des signaux
        self.signal_history = {}
        
        # Charger la configuration et les performances
        self._load_config()
        self._load_performance()
        
        # Découvrir et charger les stratégies disponibles
        self._discover_strategies()
    
    def _load_config(self) -> None:
        """Charge la configuration depuis le fichier"""
        if os.path.exists(self.config_path):
            try:
                with open(self.config_path, 'r') as f:
                    config = json.load(f)
                
                # Mettre à jour la config avec les valeurs chargées
                self.config.update(config)
                logger.info(f"Configuration chargée: {len(self.config.get('default_weights', {}))} stratégies configurées")
            except Exception as e:
                logger.error(f"Erreur lors du chargement de la configuration: {str(e)}")
    
    def _save_config(self) -> None:
        """Sauvegarde la configuration dans un fichier"""
        try:
            with open(self.config_path, 'w') as f:
                json.dump(self.config, f, indent=2)
            logger.info(f"Configuration sauvegardée dans {self.config_path}")
        except Exception as e:
            logger.error(f"Erreur lors de la sauvegarde de la configuration: {str(e)}")
    
    def _load_performance(self) -> None:
        """Charge les données de performance depuis le fichier"""
        if os.path.exists(self.performance_path):
            try:
                with open(self.performance_path, 'r') as f:
                    self.performance = json.load(f)
                logger.info(f"Performances chargées: {len(self.performance)} stratégies")
            except Exception as e:
                logger.error(f"Erreur lors du chargement des performances: {str(e)}")
                self.performance = {}
    
    def _save_performance(self) -> None:
        """Sauvegarde les données de performance dans un fichier"""
        try:
            with open(self.performance_path, 'w') as f:
                json.dump(self.performance, f, indent=2)
            logger.info("Performances sauvegardées")
        except Exception as e:
            logger.error(f"Erreur lors de la sauvegarde des performances: {str(e)}")
    
    def _discover_strategies(self) -> None:
        """
        Découvre et charge automatiquement les modules de stratégie disponibles
        """
        if not os.path.exists(self.strategies_dir):
            logger.error(f"Le répertoire des stratégies n'existe pas: {self.strategies_dir}")
            return
        
        # Liste des fichiers Python dans le répertoire des stratégies
        strategy_files = [
            f[:-3] for f in os.listdir(self.strategies_dir) 
            if f.endswith('.py') and not f.startswith('__')
        ]
        
        # Importer et initialiser chaque stratégie
        for file_name in strategy_files:
            try:
                # Construire le chemin d'importation
                import_path = f"strategies.{file_name}"
                
                # Importer le module
                module = importlib.import_module(import_path)
                
                # Chercher les classes de stratégie dans le module
                for name, obj in inspect.getmembers(module):
                    # Vérifier si c'est une classe qui contient une méthode "generate_signal"
                    if (inspect.isclass(obj) and 
                        hasattr(obj, 'generate_signal') and 
                        callable(getattr(obj, 'generate_signal'))):
                        
                        # Identifier unique pour la stratégie
                        strategy_id = f"{file_name}.{name}"
                        
                        try:
                            # Initialiser la stratégie
                            strategy_instance = obj()
                            
                            # Stocker la stratégie
                            self.strategies[strategy_id] = {
                                "id": strategy_id,
                                "name": name,
                                "module": file_name,
                                "instance": strategy_instance,
                                "description": getattr(obj, '__doc__', "No description"),
                                "params": getattr(strategy_instance, 'params', {})
                            }
                            
                            logger.info(f"Stratégie chargée: {strategy_id}")
                            
                            # Activer par défaut si présente dans la configuration
                            if strategy_id in self.config.get("default_weights", {}):
                                weight = self.config["default_weights"][strategy_id]
                                if weight > 0:
                                    self.active_strategies.append(strategy_id)
                                    logger.info(f"Stratégie activée: {strategy_id} (poids: {weight})")
                            
                        except Exception as e:
                            logger.error(f"Erreur lors de l'initialisation de la stratégie {strategy_id}: {str(e)}")
                
            except Exception as e:
                logger.error(f"Erreur lors du chargement du module {file_name}: {str(e)}")
        
        logger.info(f"Stratégies découvertes: {len(self.strategies)}, actives: {len(self.active_strategies)}")
    
    def activate_strategy(self, strategy_id: str, weight: float = 1.0) -> bool:
        """
        Active une stratégie pour l'intégration
        
        Args:
            strategy_id: Identifiant de la stratégie
            weight: Poids de la stratégie dans l'intégration
            
        Returns:
            Succès de l'activation
        """
        if strategy_id not in self.strategies:
            logger.error(f"Stratégie inconnue: {strategy_id}")
            return False
        
        # Ajouter à la liste des stratégies actives si pas déjà présente
        if strategy_id not in self.active_strategies:
            self.active_strategies.append(strategy_id)
        
        # Mettre à jour le poids
        if "default_weights" not in self.config:
            self.config["default_weights"] = {}
        
        self.config["default_weights"][strategy_id] = weight
        
        # Sauvegarder la configuration
        self._save_config()
        
        logger.info(f"Stratégie activée: {strategy_id} (poids: {weight})")
        return True
    
    def deactivate_strategy(self, strategy_id: str) -> bool:
        """
        Désactive une stratégie
        
        Args:
            strategy_id: Identifiant de la stratégie
            
        Returns:
            Succès de la désactivation
        """
        if strategy_id not in self.strategies:
            logger.error(f"Stratégie inconnue: {strategy_id}")
            return False
        
        # Retirer de la liste des stratégies actives
        if strategy_id in self.active_strategies:
            self.active_strategies.remove(strategy_id)
        
        # Mettre le poids à 0
        if "default_weights" in self.config and strategy_id in self.config["default_weights"]:
            self.config["default_weights"][strategy_id] = 0
        
        # Sauvegarder la configuration
        self._save_config()
        
        logger.info(f"Stratégie désactivée: {strategy_id}")
        return True
    
    def generate_trade_signal(self, symbol: str, data: pd.DataFrame, timeframe: str) -> Dict:
        """
        Génère un signal de trading intégré à partir des stratégies actives
        
        Args:
            symbol: Symbole de trading
            data: DataFrame avec les données OHLCV
            timeframe: Intervalle de temps
            
        Returns:
            Signal de trading intégré
        """
        if not self.active_strategies:
            return {
                "direction": "NEUTRAL",
                "signal_type": "NEUTRAL",
                "confidence": 0.0,
                "strength": 0,
                "signals": [],
                "timestamp": datetime.now().isoformat(),
                "message": "Aucune stratégie active"
            }
        
        # Exécuter toutes les stratégies actives
        signals = []
        weights = []
        
        for strategy_id in self.active_strategies:
            strategy_info = self.strategies.get(strategy_id)
            
            if not strategy_info:
                continue
            
            try:
                # Obtenir l'instance de la stratégie
                strategy = strategy_info["instance"]
                
                # Générer le signal
                signal = strategy.generate_signal(symbol, data, timeframe)
                
                # Vérifier que le signal est valide
                if isinstance(signal, dict) and "direction" in signal:
                    # Obtenir le poids de la stratégie
                    weight = self._get_strategy_weight(strategy_id, symbol)
                    
                    # Ajouter le signal et le poids
                    signals.append({
                        "strategy_id": strategy_id,
                        "signal": signal,
                        "weight": weight
                    })
                    weights.append(weight)
            
            except Exception as e:
                logger.error(f"Erreur lors de l'exécution de la stratégie {strategy_id}: {str(e)}")
        
        if not signals:
            return {
                "direction": "NEUTRAL",
                "signal_type": "NEUTRAL",
                "confidence": 0.0,
                "strength": 0,
                "signals": [],
                "timestamp": datetime.now().isoformat(),
                "message": "Aucun signal généré"
            }
        
        # Normaliser les poids
        total_weight = sum(weights)
        if total_weight > 0:
            normalized_weights = [w / total_weight for w in weights]
        else:
            normalized_weights = [1.0 / len(weights)] * len(weights)
        
        # Intégrer les signaux
        integrated_signal = self._integrate_signals(signals, normalized_weights, symbol)
        
        # Enregistrer le signal dans l'historique
        self._record_signal(symbol, integrated_signal)
        
        return integrated_signal
    
    def _get_strategy_weight(self, strategy_id: str, symbol: str) -> float:
        """
        Obtient le poids actuel d'une stratégie, adapté selon les performances
        
        Args:
            strategy_id: Identifiant de la stratégie
            symbol: Symbole de trading
            
        Returns:
            Poids de la stratégie
        """
        # Poids par défaut
        default_weight = self.config.get("default_weights", {}).get(strategy_id, 1.0)
        
        # Si l'adaptation des poids est désactivée, retourner le poids par défaut
        if not self.config.get("adaptive_weights", True):
            return default_weight
        
        # Vérifier si des données de performance sont disponibles
        if strategy_id not in self.performance:
            return default_weight
        
        # Vérifier si des données spécifiques au symbole sont disponibles
        symbol_perf = self.performance[strategy_id].get(symbol, {})
        
        # Si pas assez d'échantillons, utiliser le poids par défaut
        min_samples = self.config.get("min_samples", 10)
        if symbol_perf.get("total_signals", 0) < min_samples:
            return default_weight
        
        # Calculer le poids adaptatif basé sur le taux de réussite
        win_rate = symbol_perf.get("win_rate", 50.0)
        
        # Ajuster le poids en fonction du taux de réussite
        # Une stratégie avec 50% de réussite conserve son poids par défaut
        # Au-dessus de 50%, le poids augmente proportionnellement
        # En-dessous de 50%, le poids diminue proportionnellement
        if win_rate > 50:
            # Augmenter le poids jusqu'à 2 fois
            weight_multiplier = 1.0 + min(1.0, (win_rate - 50) / 50)
        else:
            # Diminuer le poids jusqu'à 0.5 fois
            weight_multiplier = max(0.5, 1.0 - (50 - win_rate) / 50)
        
        adaptive_weight = default_weight * weight_multiplier
        
        return adaptive_weight
    
    def _integrate_signals(self, signals: List[Dict], weights: List[float], symbol: str) -> Dict:
        """
        Intègre plusieurs signaux en un seul signal de trading
        
        Args:
            signals: Liste des signaux avec leurs poids
            weights: Poids normalisés pour chaque signal
            symbol: Symbole de trading
            
        Returns:
            Signal intégré
        """
        if not signals:
            return {
                "direction": "NEUTRAL",
                "signal_type": "NEUTRAL",
                "confidence": 0.0,
                "strength": 0,
                "timestamp": datetime.now().isoformat(),
                "signals": [],
                "symbol": symbol
            }
        
        # Compter les signaux dans chaque direction
        buy_count = sum(1 for s in signals if s["signal"]["direction"] == "BUY")
        sell_count = sum(1 for s in signals if s["signal"]["direction"] == "SELL")
        neutral_count = len(signals) - buy_count - sell_count
        
        # Calculer les scores pondérés pour chaque direction
        buy_score = sum(weights[i] * s["signal"].get("confidence", 0.5) 
                        for i, s in enumerate(signals) 
                        if s["signal"]["direction"] == "BUY")
        
        sell_score = sum(weights[i] * s["signal"].get("confidence", 0.5) 
                         for i, s in enumerate(signals) 
                         if s["signal"]["direction"] == "SELL")
        
        # Normaliser les scores
        total_score = buy_score + sell_score
        if total_score > 0:
            buy_score /= total_score
            sell_score /= total_score
        
        # Déterminer la direction finale
        direction = "NEUTRAL"
        confidence = 0.0
        
        if buy_score > sell_score:
            direction = "BUY"
            confidence = buy_score
        elif sell_score > buy_score:
            direction = "SELL"
            confidence = sell_score
        
        # Vérifier le seuil de confiance
        confidence_threshold = self.config.get("confidence_threshold", 0.5)
        if confidence < confidence_threshold:
            direction = "NEUTRAL"
            confidence = 0.0
        
        # Vérifier s'il y a consensus
        total_signals = len(signals)
        majority_threshold = self.config.get("majority_threshold", 0.6)
        
        # Boost de confiance si forte majorité
        if direction == "BUY" and buy_count / total_signals >= majority_threshold:
            consensus_boost = self.config.get("consensus_boost", 0.2)
            confidence = min(1.0, confidence + consensus_boost)
        elif direction == "SELL" and sell_count / total_signals >= majority_threshold:
            consensus_boost = self.config.get("consensus_boost", 0.2)
            confidence = min(1.0, confidence + consensus_boost)
        
        # Déterminer le type de signal
        signal_type = "NEUTRAL"
        if direction != "NEUTRAL":
            if confidence >= 0.8:
                signal_type = f"STRONG_{direction}"
            elif confidence >= 0.65:
                signal_type = f"MODERATE_{direction}"
            elif confidence >= 0.5:
                signal_type = f"WEAK_{direction}"
        
        # Calculer la force du signal (de -100 à +100)
        # 0 = neutre, positif = achat, négatif = vente
        if direction == "BUY":
            strength = int(confidence * 100)
        elif direction == "SELL":
            strength = -int(confidence * 100)
        else:
            strength = 0
        
        # Créer le signal intégré avec les détails des signaux individuels
        integrated_signal = {
            "direction": direction,
            "signal_type": signal_type,
            "confidence": float(confidence),
            "strength": strength,
            "timestamp": datetime.now().isoformat(),
            "symbol": symbol,
            "signals": [
                {
                    "strategy_id": s["strategy_id"],
                    "direction": s["signal"]["direction"],
                    "confidence": s["signal"].get("confidence", 0.5),
                    "weight": float(weights[i]),
                    "details": s["signal"].get("details", {})
                }
                for i, s in enumerate(signals)
            ],
            "stats": {
                "buy_count": buy_count,
                "sell_count": sell_count,
                "neutral_count": neutral_count,
                "total_signals": total_signals
            }
        }
        
        return integrated_signal
    
    def _record_signal(self, symbol: str, signal: Dict) -> None:
        """
        Enregistre un signal dans l'historique
        
        Args:
            symbol: Symbole de trading
            signal: Signal de trading généré
        """
        # Initialiser l'historique pour le symbole si nécessaire
        if symbol not in self.signal_history:
            self.signal_history[symbol] = []
        
        # Ajouter le signal à l'historique
        self.signal_history[symbol].append({
            "timestamp": signal["timestamp"],
            "direction": signal["direction"],
            "confidence": signal["confidence"],
            "strength": signal["strength"],
            "result": None  # Sera mis à jour plus tard avec le résultat
        })
        
        # Limiter la taille de l'historique
        max_history = 1000
        if len(self.signal_history[symbol]) > max_history:
            self.signal_history[symbol] = self.signal_history[symbol][-max_history:]
    
    def update_performance(self, symbol: str, signal_timestamp: str, 
                         actual_outcome: str, pnl: float) -> None:
        """
        Met à jour les performances des stratégies après un trade
        
        Args:
            symbol: Symbole de trading
            signal_timestamp: Timestamp du signal utilisé pour le trade
            actual_outcome: Résultat réel ('BUY', 'SELL', ou 'NEUTRAL')
            pnl: Profit/perte réalisé
        """
        # Trouver le signal correspondant
        if symbol not in self.signal_history:
            logger.warning(f"Pas d'historique disponible pour {symbol}")
            return
        
        signal_found = None
        for signal in self.signal_history[symbol]:
            if signal["timestamp"] == signal_timestamp:
                signal_found = signal
                break
        
        if not signal_found:
            logger.warning(f"Signal non trouvé pour {symbol} à {signal_timestamp}")
            return
        
        # Mettre à jour le résultat du signal
        signal_found["result"] = {
            "actual_outcome": actual_outcome,
            "pnl": pnl,
            "updated_at": datetime.now().isoformat()
        }
        
        # Récupérer le signal complet (avec les détails des stratégies)
        full_signal = None
        for s in self.signal_history[symbol]:
            if s["timestamp"] == signal_timestamp and isinstance(s, dict) and "signals" in s:
                full_signal = s
                break
        
        if not full_signal or "signals" not in full_signal:
            logger.warning(f"Détails du signal non trouvés pour {symbol} à {signal_timestamp}")
            return
        
        # Mettre à jour les performances de chaque stratégie
        for strategy_signal in full_signal["signals"]:
            strategy_id = strategy_signal["strategy_id"]
            strategy_direction = strategy_signal["direction"]
            
            # Déterminer si la stratégie avait raison
            success = (strategy_direction == actual_outcome and strategy_direction != "NEUTRAL")
            
            # Initialiser les statistiques pour cette stratégie si nécessaire
            if strategy_id not in self.performance:
                self.performance[strategy_id] = {}
            
            if symbol not in self.performance[strategy_id]:
                self.performance[strategy_id][symbol] = {
                    "total_signals": 0,
                    "correct_signals": 0,
                    "incorrect_signals": 0,
                    "neutral_signals": 0,
                    "total_pnl": 0.0,
                    "win_rate": 0.0,
                    "avg_pnl": 0.0,
                    "last_updated": None
                }
            
            stats = self.performance[strategy_id][symbol]
            
            # Mettre à jour les statistiques
            stats["total_signals"] += 1
            
            if strategy_direction == "NEUTRAL":
                stats["neutral_signals"] += 1
            elif success:
                stats["correct_signals"] += 1
                stats["total_pnl"] += pnl
            else:
                stats["incorrect_signals"] += 1
                stats["total_pnl"] += pnl
            
            # Calculer le taux de réussite
            non_neutral_signals = stats["correct_signals"] + stats["incorrect_signals"]
            if non_neutral_signals > 0:
                stats["win_rate"] = stats["correct_signals"] / non_neutral_signals * 100
            
            # Calculer le PnL moyen
            if stats["total_signals"] > 0:
                stats["avg_pnl"] = stats["total_pnl"] / stats["total_signals"]
            
            stats["last_updated"] = datetime.now().isoformat()
        
        # Sauvegarder les performances mises à jour
        self._save_performance()
    
    def get_performance_metrics(self) -> Dict:
        """
        Récupère les métriques de performance de toutes les stratégies
        
        Returns:
            Métriques de performance
        """
        metrics = {
            "strategies": {},
            "active_strategies": len(self.active_strategies),
            "total_strategies": len(self.strategies),
            "last_updated": datetime.now().isoformat()
        }
        
        # Agréger les métriques pour chaque stratégie
        for strategy_id, strategy_perf in self.performance.items():
            # Calculer les moyennes sur tous les symboles
            total_signals = 0
            correct_signals = 0
            incorrect_signals = 0
            neutral_signals = 0
            total_pnl = 0.0
            
            for symbol, stats in strategy_perf.items():
                total_signals += stats.get("total_signals", 0)
                correct_signals += stats.get("correct_signals", 0)
                incorrect_signals += stats.get("incorrect_signals", 0)
                neutral_signals += stats.get("neutral_signals", 0)
                total_pnl += stats.get("total_pnl", 0.0)
            
            # Calculer le taux de réussite global
            non_neutral_signals = correct_signals + incorrect_signals
            win_rate = 0.0
            if non_neutral_signals > 0:
                win_rate = correct_signals / non_neutral_signals * 100
            
            # Calculer le PnL moyen
            avg_pnl = 0.0
            if total_signals > 0:
                avg_pnl = total_pnl / total_signals
            
            # Stocker les métriques pour cette stratégie
            metrics["strategies"][strategy_id] = {
                "win_rate": win_rate,
                "total_signals": total_signals,
                "correct_signals": correct_signals,
                "incorrect_signals": incorrect_signals,
                "neutral_signals": neutral_signals,
                "total_pnl": total_pnl,
                "avg_pnl": avg_pnl,
                "by_symbol": {s: stats for s, stats in strategy_perf.items()}
            }
            
            # Ajouter des informations sur la stratégie
            if strategy_id in self.strategies:
                metrics["strategies"][strategy_id]["name"] = self.strategies[strategy_id]["name"]
                metrics["strategies"][strategy_id]["module"] = self.strategies[strategy_id]["module"]
                metrics["strategies"][strategy_id]["active"] = strategy_id in self.active_strategies
                metrics["strategies"][strategy_id]["weight"] = self.config.get("default_weights", {}).get(strategy_id, 0.0)
        
        return metrics
    
    def get_available_strategies(self) -> List[Dict]:
        """
        Récupère la liste des stratégies disponibles
        
        Returns:
            Liste des stratégies avec leurs informations
        """
        available_strategies = []
        
        for strategy_id, strategy_info in self.strategies.items():
            # Créer un dictionnaire d'informations sur la stratégie
            strategy_data = {
                "id": strategy_id,
                "name": strategy_info["name"],
                "module": strategy_info["module"],
                "description": strategy_info["description"],
                "params": strategy_info["params"],
                "active": strategy_id in self.active_strategies,
                "weight": self.config.get("default_weights", {}).get(strategy_id, 0.0)
            }
            
            # Ajouter les données de performance si disponibles
            if strategy_id in self.performance:
                # Calculer les métriques agrégées
                total_signals = 0
                correct_signals = 0
                for symbol, stats in self.performance[strategy_id].items():
                    total_signals += stats.get("total_signals", 0)
                    correct_signals += stats.get("correct_signals", 0)
                
                win_rate = 0.0
                if total_signals > 0:
                    win_rate = correct_signals / total_signals * 100
                
                strategy_data["performance"] = {
                    "win_rate": win_rate,
                    "total_signals": total_signals
                }
            
            available_strategies.append(strategy_data)
        
        return available_strategies
    
    def update_strategy_params(self, strategy_id: str, params: Dict) -> bool:
        """
        Met à jour les paramètres d'une stratégie
        
        Args:
            strategy_id: Identifiant de la stratégie
            params: Nouveaux paramètres
            
        Returns:
            Succès de la mise à jour
        """
        if strategy_id not in self.strategies:
            logger.error(f"Stratégie inconnue: {strategy_id}")
            return False
        
        try:
            # Obtenir l'instance de la stratégie
            strategy = self.strategies[strategy_id]["instance"]
            
            # Mettre à jour les paramètres
            if hasattr(strategy, 'params'):
                strategy.params.update(params)
                
                # Mettre à jour dans le dictionnaire pour la persistance
                self.strategies[strategy_id]["params"] = strategy.params
                
                logger.info(f"Paramètres mis à jour pour {strategy_id}: {params}")
                return True
            else:
                logger.warning(f"La stratégie {strategy_id} n'a pas d'attribut 'params'")
                return False
            
        except Exception as e:
            logger.error(f"Erreur lors de la mise à jour des paramètres de {strategy_id}: {str(e)}")
            return False
    
    def update_configuration(self, new_config: Dict) -> Dict:
        """
        Met à jour la configuration de l'intégrateur
        
        Args:
            new_config: Nouvelle configuration
            
        Returns:
            Configuration mise à jour
        """
        # Mettre à jour la configuration avec les nouvelles valeurs
        for key, value in new_config.items():
            if key in self.config:
                self.config[key] = value
            elif key == "default_weights":
                # Cas spécial pour les poids
                if "default_weights" not in self.config:
                    self.config["default_weights"] = {}
                
                # Mettre à jour seulement les stratégies existantes
                for strategy_id, weight in value.items():
                    if strategy_id in self.strategies:
                        self.config["default_weights"][strategy_id] = weight
        
        # Sauvegarder la configuration mise à jour
        self._save_config()
        
        return self.config
    
    def get_strategy_signals(self, symbol: str, limit: int = 10) -> List[Dict]:
        """
        Récupère l'historique des signaux pour un symbole donné
        
        Args:
            symbol: Symbole de trading
            limit: Nombre maximum de signaux à renvoyer
            
        Returns:
            Liste des derniers signaux générés
        """
        if symbol not in self.signal_history:
            return []
        
        # Renvoyer les derniers signaux
        return self.signal_history[symbol][-limit:]
    
    def get_strategy(self, strategy_id: str) -> Optional[Dict]:
        """
        Récupère les informations d'une stratégie spécifique
        
        Args:
            strategy_id: Identifiant de la stratégie
            
        Returns:
            Informations sur la stratégie ou None si non trouvée
        """
        if strategy_id not in self.strategies:
            return None
        
        strategy_info = self.strategies[strategy_id].copy()
        
        # Ne pas renvoyer l'instance de la stratégie (non sérialisable)
        if "instance" in strategy_info:
            del strategy_info["instance"]
        
        # Ajouter des informations de performance si disponibles
        if strategy_id in self.performance:
            strategy_info["performance"] = self.performance[strategy_id]
        
        # Ajouter le statut d'activation
        strategy_info["active"] = strategy_id in self.active_strategies
        strategy_info["weight"] = self.config.get("default_weights", {}).get(strategy_id, 0.0)
        
        return strategy_info
    
    def reset_performance_stats(self, strategy_id: Optional[str] = None, symbol: Optional[str] = None) -> bool:
        """
        Réinitialise les statistiques de performance
        
        Args:
            strategy_id: ID de la stratégie (toutes les stratégies si None)
            symbol: Symbole spécifique (tous les symboles si None)
            
        Returns:
            Succès de la réinitialisation
        """
        try:
            if strategy_id is None:
                # Réinitialiser toutes les statistiques
                if symbol is None:
                    self.performance = {}
                else:
                    # Réinitialiser uniquement pour un symbole spécifique
                    for strat_id in self.performance:
                        if symbol in self.performance[strat_id]:
                            del self.performance[strat_id][symbol]
            else:
                # Réinitialiser uniquement pour une stratégie spécifique
                if strategy_id in self.performance:
                    if symbol is None:
                        self.performance[strategy_id] = {}
                    elif symbol in self.performance[strategy_id]:
                        del self.performance[strategy_id][symbol]
            
            # Sauvegarder les performances mises à jour
            self._save_performance()
            
            return True
            
        except Exception as e:
            logger.error(f"Erreur lors de la réinitialisation des statistiques: {str(e)}")
            return False