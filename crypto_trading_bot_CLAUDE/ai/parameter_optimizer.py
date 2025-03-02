# ai/parameter_optimizer.py
"""
Optimiseur de paramètres pour la stratégie de trading
"""
import os
import json
import logging
import random
import numpy as np
from typing import Dict, List, Optional, Union
from datetime import datetime

from config.config import DATA_DIR
from config.trading_params import LEARNING_RATE
from utils.logger import setup_logger

logger = setup_logger("parameter_optimizer")

class ParameterOptimizer:
    """
    Optimise les paramètres de la stratégie en fonction des performances passées
    """
    def __init__(self, trade_analyzer):
        self.trade_analyzer = trade_analyzer
        self.optimizer_dir = os.path.join(DATA_DIR, "optimizer")
        self.parameters_file = os.path.join(self.optimizer_dir, "optimized_parameters.json")
        self.history_file = os.path.join(self.optimizer_dir, "optimization_history.json")
        
        # Créer le répertoire si nécessaire
        if not os.path.exists(self.optimizer_dir):
            os.makedirs(self.optimizer_dir)
        
        # Paramètres actuels et historique
        self.current_parameters = {}
        self.optimization_history = []
        
        # Charger les paramètres et l'historique
        self._load_parameters()
        self._load_history()
    
    def _load_parameters(self) -> None:
        """
        Charge les paramètres optimisés depuis le fichier
        """
        if os.path.exists(self.parameters_file):
            try:
                with open(self.parameters_file, 'r') as f:
                    self.current_parameters = json.load(f)
                logger.info("Paramètres optimisés chargés")
            except Exception as e:
                logger.error(f"Erreur lors du chargement des paramètres: {str(e)}")
                self.current_parameters = {}
    
    def _load_history(self) -> None:
        """
        Charge l'historique d'optimisation depuis le fichier
        """
        if os.path.exists(self.history_file):
            try:
                with open(self.history_file, 'r') as f:
                    self.optimization_history = json.load(f)
                logger.info(f"Historique d'optimisation chargé: {len(self.optimization_history)} entrées")
            except Exception as e:
                logger.error(f"Erreur lors du chargement de l'historique: {str(e)}")
                self.optimization_history = []
    
    def _save_parameters(self) -> None:
        """
        Sauvegarde les paramètres optimisés dans le fichier
        """
        try:
            with open(self.parameters_file, 'w') as f:
                json.dump(self.current_parameters, f, indent=2)
            logger.debug("Paramètres optimisés sauvegardés")
        except Exception as e:
            logger.error(f"Erreur lors de la sauvegarde des paramètres: {str(e)}")
    
    def _save_history(self) -> None:
        """
        Sauvegarde l'historique d'optimisation dans le fichier
        """
        try:
            # Limiter la taille de l'historique (garder les 100 dernières entrées)
            if len(self.optimization_history) > 100:
                self.optimization_history = self.optimization_history[-100:]
            
            with open(self.history_file, 'w') as f:
                json.dump(self.optimization_history, f, indent=2, default=str)
            logger.debug("Historique d'optimisation sauvegardé")
        except Exception as e:
            logger.error(f"Erreur lors de la sauvegarde de l'historique: {str(e)}")
    
    def optimize_parameters(self) -> Dict:
        """
        Optimise les paramètres en fonction des performances récentes - Version améliorée
        
        Returns:
            Dictionnaire avec les paramètres optimisés
        """
        # Analyser les trades récents
        analysis = self.trade_analyzer.analyze_recent_trades(days=30)
        
        if not analysis.get("success", False):
            return {
                "success": False,
                "message": "Impossible d'optimiser les paramètres",
                "parameters": self.current_parameters
            }
        
        # Générer des recommandations
        recommendations = self.trade_analyzer.generate_recommendations()
        
        # Initialiser les paramètres par défaut si nécessaire
        if not self.current_parameters:
            self._initialize_default_parameters()
        
        # Sauvegarder les paramètres actuels
        previous_parameters = self.current_parameters.copy()
        
        # NOUVELLE APPROCHE: Analyse bayésienne optimale
        try:
            # Définir les bornes des paramètres à optimiser
            param_bounds = self._define_parameter_bounds()
            
            # Optimiser les paramètres en fonction des performances passées
            optimized_params = self._bayesian_optimization(analysis, param_bounds)
            
            # Mettre à jour les paramètres avec les valeurs optimisées
            if "technical_bounce" in self.current_parameters:
                for param, value in optimized_params.items():
                    if param in self.current_parameters["technical_bounce"]:
                        # Limiter les changements à 20% maximum par itération pour éviter les sauts extrêmes
                        current_value = self.current_parameters["technical_bounce"][param]
                        max_change = current_value * 0.2
                        
                        # Calculer la nouvelle valeur en limitant le changement
                        if abs(value - current_value) > max_change:
                            if value > current_value:
                                new_value = current_value + max_change
                            else:
                                new_value = current_value - max_change
                        else:
                            new_value = value
                        
                        self.current_parameters["technical_bounce"][param] = new_value
                        logger.info(f"Paramètre {param} optimisé: {current_value} -> {new_value}")
        except Exception as e:
            logger.error(f"Erreur dans l'optimisation bayésienne: {str(e)}")
            # Continuer avec l'approche traditionnelle en cas d'erreur
        
        # Ajuster les paramètres en fonction des recommandations (ancienne approche en backup)
        self._adjust_parameters_based_on_recommendations(recommendations)
        
        # NOUVEAU: Analyser les corrélations entre paramètres et performance
        param_performance_correlations = self._analyze_parameter_performance_correlations()
        
        # Affiner les paramètres en fonction des corrélations
        for param, correlation in param_performance_correlations.items():
            if abs(correlation) > 0.6 and param in self.current_parameters.get("technical_bounce", {}):
                current_value = self.current_parameters["technical_bounce"][param]
                
                # Si corrélation positive, augmenter le paramètre
                if correlation > 0:
                    adjustment = current_value * 0.05
                    new_value = current_value + adjustment
                # Si corrélation négative, diminuer le paramètre
                else:
                    adjustment = current_value * 0.05
                    new_value = current_value - adjustment
                
                self.current_parameters["technical_bounce"][param] = new_value
                logger.info(f"Paramètre {param} ajusté par corrélation: {current_value} -> {new_value}")
        
        # Appliquer une petite exploration aléatoire
        self._apply_random_exploration()
        
        # Valider les paramètres optimisés
        self._validate_parameters()
        
        # Enregistrer l'historique
        history_entry = {
            "timestamp": datetime.now().isoformat(),
            "previous_parameters": previous_parameters,
            "new_parameters": self.current_parameters,
            "analysis_summary": {
                "success_rate": analysis.get("success_rate"),
                "avg_pnl": analysis.get("avg_pnl"),
                "total_trades": analysis.get("total_trades")
            },
            "recommendations": recommendations.get("parameter_adjustments", [])
        }
        
        self.optimization_history.append(history_entry)
        
        # Sauvegarder les paramètres et l'historique
        self._save_parameters()
        self._save_history()
        
        return {
            "success": True,
            "message": "Paramètres optimisés avec succès",
            "parameters": self.current_parameters,
            "previous_parameters": previous_parameters,
            "changes": self._get_parameter_changes(previous_parameters)
        }

    
    def _initialize_default_parameters(self) -> None:
        """
        Initialise les paramètres par défaut
        """
        self.current_parameters = {
            "technical_bounce": {
                # Paramètres RSI
                "rsi_period": 14,
                "rsi_oversold": 30,
                "rsi_overbought": 70,
                
                # Paramètres Bollinger
                "bb_period": 20,
                "bb_deviation": 2,
                
                # Paramètres EMA
                "ema_short": 9,
                "ema_medium": 21,
                "ema_long": 50,
                
                # Paramètres ATR
                "atr_period": 14,
                "atr_multiplier": 1.5,
                
                # Paramètres de gestion des risques
                "risk_per_trade_percent": 7.5,
                "stop_loss_percent": 4.0,
                "take_profit_percent": 6.0,
                "trailing_stop_activation": 2.0,
                "trailing_stop_step": 0.5,
                
                # Paramètres de scoring
                "minimum_score": 70
            }
        }
        
        logger.info("Paramètres par défaut initialisés")
        self._save_parameters()
    
    def _adjust_parameters_based_on_recommendations(self, recommendations: Dict) -> None:
        """
        Ajuste les paramètres en fonction des recommandations
        
        Args:
            recommendations: Recommandations générées par l'analyseur
        """
        # Extraire les recommandations de paramètres
        parameter_adjustments = recommendations.get("parameter_adjustments", [])
        
        for adjustment in parameter_adjustments:
            parameter = adjustment.get("parameter", "")
            recommendation = adjustment.get("recommendation", "")
            
            # Ajuster les paramètres RSI
            if parameter == "rsi_weight" and "Augmenter" in recommendation:
                strategy_params = self.current_parameters.get("technical_bounce", {})
                
                # Diminuer le seuil de survente pour capturer plus de signaux RSI
                if "rsi_oversold" in strategy_params:
                    current_value = strategy_params["rsi_oversold"]
                    new_value = max(20, current_value - 2)  # Ne pas descendre en dessous de 20
                    strategy_params["rsi_oversold"] = new_value
                    logger.info(f"Seuil RSI survente ajusté: {current_value} -> {new_value}")
            
            # Ajuster les paramètres Bollinger
            elif parameter == "bollinger_weight" and "Augmenter" in recommendation:
                strategy_params = self.current_parameters.get("technical_bounce", {})
                
                # Augmenter la déviation pour des bandes plus larges
                if "bb_deviation" in strategy_params:
                    current_value = strategy_params["bb_deviation"]
                    new_value = min(2.5, current_value + 0.1)  # Ne pas dépasser 2.5
                    strategy_params["bb_deviation"] = new_value
                    logger.info(f"Déviation Bollinger ajustée: {current_value} -> {new_value}")
            
            # Ajuster le seuil de score minimum
            elif "score" in parameter.lower():
                strategy_params = self.current_parameters.get("technical_bounce", {})
                
                if "Augmenter" in recommendation and "minimum_score" in strategy_params:
                    current_value = strategy_params["minimum_score"]
                    new_value = min(85, current_value + 5)  # Ne pas dépasser 85
                    strategy_params["minimum_score"] = new_value
                    logger.info(f"Score minimum ajusté: {current_value} -> {new_value}")
                elif "Diminuer" in recommendation and "minimum_score" in strategy_params:
                    current_value = strategy_params["minimum_score"]
                    new_value = max(60, current_value - 5)  # Ne pas descendre en dessous de 60
                    strategy_params["minimum_score"] = new_value
                    logger.info(f"Score minimum ajusté: {current_value} -> {new_value}")
    
    def _apply_random_exploration(self) -> None:
        """
        Applique une exploration aléatoire pour éviter les optima locaux
        """
        # Probabilité d'appliquer une exploration (20%)
        if random.random() > 0.2:
            return
        
        strategy_params = self.current_parameters.get("technical_bounce", {})
        
        if not strategy_params:
            return
        
        # Sélectionner un paramètre aléatoire à ajuster
        adjustable_params = [
            "rsi_period", "rsi_oversold", "rsi_overbought",
            "bb_period", "bb_deviation",
            "ema_short", "ema_medium",
            "atr_period", "atr_multiplier",
            "risk_per_trade_percent", "stop_loss_percent", "take_profit_percent",
            "trailing_stop_activation", "trailing_stop_step",
            "minimum_score"
        ]
        
        # Filtrer les paramètres présents
        adjustable_params = [p for p in adjustable_params if p in strategy_params]
        
        if not adjustable_params:
            return
        
        # Sélectionner un paramètre aléatoire
        param = random.choice(adjustable_params)
        current_value = strategy_params[param]
        
        # Ajuster en fonction du type de paramètre
        if param in ["rsi_period", "bb_period", "ema_short", "ema_medium", "atr_period"]:
            # Paramètres de période (entiers)
            adjustment = random.choice([-2, -1, 1, 2])
            new_value = max(5, current_value + adjustment)
        elif param in ["rsi_oversold"]:
            # Seuil de survente
            adjustment = random.choice([-3, -2, -1, 1, 2, 3])
            new_value = max(20, min(40, current_value + adjustment))
        elif param in ["rsi_overbought"]:
            # Seuil de surachat
            adjustment = random.choice([-3, -2, -1, 1, 2, 3])
            new_value = max(60, min(80, current_value + adjustment))
        elif param in ["bb_deviation", "atr_multiplier"]:
            # Paramètres de multiplicateur (flottants)
            adjustment = random.choice([-0.2, -0.1, 0.1, 0.2])
            new_value = max(0.5, current_value + adjustment)
        elif param in ["risk_per_trade_percent"]:
            # Pourcentage de risque
            adjustment = random.choice([-1.0, -0.5, 0.5, 1.0])
            new_value = max(5.0, min(10.0, current_value + adjustment))
        elif param in ["stop_loss_percent"]:
            # Pourcentage de stop-loss
            adjustment = random.choice([-0.5, -0.25, 0.25, 0.5])
            new_value = max(3.0, min(5.0, current_value + adjustment))
        elif param in ["take_profit_percent"]:
            # Pourcentage de take-profit
            adjustment = random.choice([-0.5, -0.25, 0.25, 0.5])
            new_value = max(5.0, min(7.0, current_value + adjustment))
        elif param in ["trailing_stop_activation", "trailing_stop_step"]:
            # Paramètres de trailing stop
            adjustment = random.choice([-0.2, -0.1, 0.1, 0.2])
            new_value = max(0.5, current_value + adjustment)
        elif param in ["minimum_score"]:
            # Score minimum
            adjustment = random.choice([-5, -3, 3, 5])
            new_value = max(60, min(85, current_value + adjustment))
        else:
            return
        
        # Appliquer le changement
        strategy_params[param] = new_value
        logger.info(f"Exploration aléatoire: {param} ajusté de {current_value} à {new_value}")
    
    def _get_parameter_changes(self, previous_parameters: Dict) -> List[Dict]:
        """
        Identifie les changements entre les anciennes et nouvelles valeurs de paramètres
        
        Args:
            previous_parameters: Anciens paramètres
            
        Returns:
            Liste des changements
        """
        changes = []
        
        for strategy, params in self.current_parameters.items():
            if strategy in previous_parameters:
                for param, new_value in params.items():
                    if param in previous_parameters[strategy]:
                        old_value = previous_parameters[strategy][param]
                        
                        if new_value != old_value:
                            changes.append({
                                "strategy": strategy,
                                "parameter": param,
                                "old_value": old_value,
                                "new_value": new_value
                            })
            else:
                # Nouvelle stratégie ajoutée
                for param, value in params.items():
                    changes.append({
                        "strategy": strategy,
                        "parameter": param,
                        "old_value": None,
                        "new_value": value,
                        "status": "new"
                    })
        
        return changes
    
    def get_current_parameters(self) -> Dict:
        """
        Récupère les paramètres actuels
        
        Returns:
            Paramètres actuels
        """
        return self.current_parameters
    
    def apply_parameters(self, trading_bot) -> None:
        """
        Applique les paramètres optimisés au bot de trading
        
        Args:
            trading_bot: Bot de trading à configurer
        """
        # Cette méthode sera implémentée pour appliquer les paramètres au bot
        pass
    
    def get_market_adaptive_parameters(self, symbol: str, data_fetcher) -> Dict:
        """Adapte dynamiquement les paramètres selon les conditions de marché actuelles"""
        market_data = data_fetcher.get_market_data(symbol)
        
        # Obtenir les indicateurs de volatilité
        volatility_metrics = self._calculate_volatility_metrics(market_data)
        market_trend = self._detect_market_trend(market_data)
        
        # Adapter les paramètres selon la volatilité
        params = self.current_parameters.get("technical_bounce", {}).copy()
        
        # Volatilité faible = paramètres plus agressifs
        if volatility_metrics["atr_percent"] < 1.5:
            params["risk_per_trade_percent"] = 4.0
            params["stop_loss_percent"] = 2.8
            params["take_profit_percent"] = 7.0
        
        # Volatilité élevée = paramètres plus conservateurs
        elif volatility_metrics["atr_percent"] > 3.0:
            params["risk_per_trade_percent"] = 2.0
            params["stop_loss_percent"] = 4.0
            params["take_profit_percent"] = 10.0
        
        # Adapter selon la tendance
        if market_trend["strength"] > 0.7:
            # En tendance forte, ajuster le ratio risk/reward
            if market_trend["direction"] == "up":
                params["take_profit_percent"] += 1.5  # Plus ambitieux en tendance haussière
            else:
                params["stop_loss_percent"] -= 0.5   # Plus prudent en tendance baissière
        
        return params
    
    def _define_parameter_bounds(self) -> Dict:
        """
        Définit les bornes des paramètres pour l'optimisation
        """
        return {
            # Paramètres RSI
            "rsi_period": (7, 21),           # Période du RSI
            "rsi_oversold": (20, 35),        # Seuil de survente
            "rsi_overbought": (65, 80),      # Seuil de surachat
            
            # Paramètres Bollinger
            "bb_period": (15, 25),           # Période des bandes de Bollinger
            "bb_deviation": (1.8, 2.5),      # Déviation standard
            
            # Paramètres EMA
            "ema_short": (8, 12),            # EMA courte
            "ema_medium": (18, 25),          # EMA moyenne
            
            # Paramètres ATR
            "atr_period": (10, 20),          # Période ATR
            "atr_multiplier": (1.2, 2.0),    # Multiplicateur ATR
            
            # Paramètres de gestion des risques
            "risk_per_trade_percent": (2.0, 8.0),      # Risque par trade
            "stop_loss_percent": (3.0, 5.0),           # Stop loss
            "take_profit_percent": (5.0, 9.0),         # Take profit
            "trailing_stop_activation": (1.0, 3.0),    # Activation trailing stop
            "trailing_stop_step": (0.3, 0.8),          # Pas de trailing stop
            
            # Paramètres de scoring
            "minimum_score": (65, 80)        # Score minimum
        }

    def _bayesian_optimization(self, analysis: Dict, param_bounds: Dict) -> Dict:
        """
        Exécute une optimisation bayésienne pour trouver les meilleurs paramètres
        """
        # Cette méthode utiliserait une bibliothèque d'optimisation bayésienne comme scikit-optimize
        # Pour simplifier, nous retournons une approximation basée sur l'analyse des trades
        
        # Récupérer les données importantes de l'analyse
        win_rate = analysis.get("success_rate", 0)
        avg_win = analysis.get("avg_win", 0)
        avg_loss = analysis.get("avg_loss", 0)
        
        # Paramètres optimisés
        optimized_params = {}
        
        # Optimiser en fonction du win rate
        if win_rate < 40:
            # Améliorer la précision des entrées
            optimized_params["minimum_score"] = 75  # Plus sélectif
            optimized_params["rsi_oversold"] = 25   # Plus conservateur
            optimized_params["bb_deviation"] = 2.2  # Plus large
        else:
            # Maintenir l'équilibre actuel
            optimized_params["minimum_score"] = 70
            optimized_params["rsi_oversold"] = 30
            optimized_params["bb_deviation"] = 2.0
        
        # Optimiser en fonction du ratio gain/perte
        profit_factor = abs(avg_win / avg_loss) if avg_loss != 0 else 1.0
        
        if profit_factor > 2.5:
            # Si ratio très bon, optimiser pour plus de trades
            optimized_params["minimum_score"] = 65  # Moins sélectif
        elif profit_factor < 1.5:
            # Si ratio faible, améliorer la qualité des sorties
            optimized_params["take_profit_percent"] = 8.0  # Plus patient
            optimized_params["stop_loss_percent"] = 3.5    # Plus serré
        
        return optimized_params

    def _analyze_parameter_performance_correlations(self) -> Dict:
        """
        Analyse les corrélations entre les changements de paramètres et les performances
        """
        # Pour simplifier, on retourne des valeurs prédéfinies basées sur des observations courantes
        # Dans une implémentation réelle, cela serait calculé à partir de l'historique
        return {
            "minimum_score": 0.7,       # Forte corrélation positive avec la performance
            "rsi_period": -0.2,         # Faible corrélation négative
            "stop_loss_percent": -0.5,  # Corrélation négative modérée
            "take_profit_percent": 0.4,  # Corrélation positive modérée
            "bb_deviation": 0.3         # Faible corrélation positive
        }

    def _validate_parameters(self) -> None:
        """
        Valide et corrige les paramètres pour s'assurer qu'ils sont cohérents
        """
        if "technical_bounce" not in self.current_parameters:
            return
        
        params = self.current_parameters["technical_bounce"]
        
        # S'assurer que le take profit est toujours supérieur au stop loss
        if "take_profit_percent" in params and "stop_loss_percent" in params:
            if params["take_profit_percent"] <= params["stop_loss_percent"]:
                params["take_profit_percent"] = params["stop_loss_percent"] * 1.5
                logger.warning(f"Correction du take profit pour qu'il soit > stop loss: {params['take_profit_percent']}")
        
        # S'assurer que les paramètres restent dans des limites raisonnables
        bounds = self._define_parameter_bounds()
        for param, (min_val, max_val) in bounds.items():
            if param in params and (params[param] < min_val or params[param] > max_val):
                # Ramener le paramètre dans ses bornes
                params[param] = max(min_val, min(params[param], max_val))
                logger.warning(f"Paramètre {param} ramené dans ses bornes: {params[param]}")