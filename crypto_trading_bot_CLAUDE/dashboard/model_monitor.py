"""
Système de monitoring avancé pour visualiser et analyser les performances du modèle
et les décisions de trading en temps réel
"""
import os
import json
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Union, Optional, Any
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.dates import DateFormatter
import seaborn as sns
from collections import defaultdict
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import base64
from io import BytesIO

from config.config import DATA_DIR
from utils.logger import setup_logger

logger = setup_logger("model_monitor")

class ModelMonitor:
    """
    Système de monitoring pour visualiser et analyser les performances du modèle LSTM
    et les décisions de trading en temps réel
    """
    def __init__(self, model=None, data_dir: str = None):
        """
        Initialise le système de monitoring
        
        Args:
            model: Modèle à monitorer (LSTM ou autre)
            data_dir: Répertoire de données
        """
        self.model = model
        self.data_dir = data_dir or os.path.join(DATA_DIR, "monitoring")
        
        # Créer le répertoire si nécessaire
        os.makedirs(self.data_dir, exist_ok=True)
        
        # Historique des prédictions
        self.prediction_history = []
        
        # Historique des trades exécutés
        self.trade_history = []
        
        # Performances du modèle
        self.model_performance = {
            "accuracy": [],
            "precision": [],
            "recall": [],
            "f1_score": [],
            "timestamps": []
        }
        
        # Logs hiérarchiques
        self.logs = {
            "error": [],
            "warning": [],
            "info": [],
            "debug": []
        }
        
        # Attributions de performance
        self.performance_attribution = {
            "model_contribution": [],
            "technical_contribution": [],
            "market_contribution": [],
            "timestamps": []
        }
        
        # Charger les données existantes
        self._load_data()
    
    def record_prediction(self, symbol: str, prediction: Dict, actual_data: Optional[Dict] = None,
                      timestamp: str = None) -> None:
        """
        Enregistre une prédiction du modèle
        
        Args:
            symbol: Paire de trading
            prediction: Prédiction du modèle
            actual_data: Données réelles (si disponibles)
            timestamp: Horodatage de la prédiction
        """
        if timestamp is None:
            timestamp = datetime.now().isoformat()
        
        # Ajouter à l'historique des prédictions
        prediction_record = {
            "symbol": symbol,
            "timestamp": timestamp,
            "prediction": prediction,
            "actual": actual_data
        }
        
        self.prediction_history.append(prediction_record)
        
        # Limiter la taille de l'historique
        if len(self.prediction_history) > 10000:
            self.prediction_history = self.prediction_history[-10000:]
        
        # Sauvegarder périodiquement
        if len(self.prediction_history) % 100 == 0:
            self._save_data()
    
    def record_trade(self, trade_data: Dict) -> None:
        """
        Enregistre un trade exécuté
        
        Args:
            trade_data: Données du trade
        """
        # Ajouter à l'historique des trades
        self.trade_history.append(trade_data)
        
        # Limiter la taille de l'historique
        if len(self.trade_history) > 1000:
            self.trade_history = self.trade_history[-1000:]
        
        # Sauvegarder
        self._save_data()
    
    def update_model_performance(self, metrics: Dict, timestamp: str = None) -> None:
        """
        Met à jour les métriques de performance du modèle
        
        Args:
            metrics: Métriques de performance
            timestamp: Horodatage de l'évaluation
        """
        if timestamp is None:
            timestamp = datetime.now().isoformat()
        
        # Ajouter les métriques
        self.model_performance["accuracy"].append(metrics.get("accuracy", 0))
        self.model_performance["precision"].append(metrics.get("precision", 0))
        self.model_performance["recall"].append(metrics.get("recall", 0))
        self.model_performance["f1_score"].append(metrics.get("f1_score", 0))
        self.model_performance["timestamps"].append(timestamp)
        
        # Limiter la taille des historiques
        max_history = 1000
        for key in self.model_performance:
            if len(self.model_performance[key]) > max_history:
                self.model_performance[key] = self.model_performance[key][-max_history:]
        
        # Sauvegarder
        self._save_data()
    
    def add_log(self, level: str, message: str, context: Optional[Dict] = None,
             timestamp: str = None) -> None:
        """
        Ajoute une entrée au journal hiérarchique
        
        Args:
            level: Niveau de log (error, warning, info, debug)
            message: Message de log
            context: Contexte additionnel
            timestamp: Horodatage du log
        """
        if timestamp is None:
            timestamp = datetime.now().isoformat()
        
        if level not in self.logs:
            level = "info"
        
        # Créer l'entrée de log
        log_entry = {
            "timestamp": timestamp,
            "message": message,
            "context": context or {}
        }
        
        self.logs[level].append(log_entry)
        
        # Limiter la taille des logs
        max_logs = 1000
        self.logs[level] = self.logs[level][-max_logs:]
        
        # Sauvegarder périodiquement
        if sum(len(logs) for logs in self.logs.values()) % 100 == 0:
            self._save_logs()
    
    def update_performance_attribution(self, model_contrib: float, technical_contrib: float,
                                   market_contrib: float, timestamp: str = None) -> None:
        """
        Met à jour l'attribution de performance entre le modèle et les règles classiques
        
        Args:
            model_contrib: Contribution du modèle (0-1)
            technical_contrib: Contribution des indicateurs techniques (0-1)
            market_contrib: Contribution des conditions de marché (0-1)
            timestamp: Horodatage de l'évaluation
        """
        if timestamp is None:
            timestamp = datetime.now().isoformat()
        
        # Ajouter les attributions
        self.performance_attribution["model_contribution"].append(model_contrib)
        self.performance_attribution["technical_contribution"].append(technical_contrib)
        self.performance_attribution["market_contribution"].append(market_contrib)
        self.performance_attribution["timestamps"].append(timestamp)
        
        # Limiter la taille des historiques
        max_history = 1000
        for key in self.performance_attribution:
            if len(self.performance_attribution[key]) > max_history:
                self.performance_attribution[key] = self.performance_attribution[key][-max_history:]
        
        # Sauvegarder
        self._save_data()
    
    def _save_data(self) -> None:
        """Sauvegarde les données de monitoring"""
        try:
            # Sauvegarder l'historique des prédictions
            predictions_path = os.path.join(self.data_dir, "prediction_history.json")
            with open(predictions_path, 'w') as f:
                # Convertir en format sérialisable
                serializable_predictions = []
                
                for pred in self.prediction_history[-1000:]:  # Limiter à 1000 entrées pour la sauvegarde
                    serializable_pred = {
                        "symbol": pred["symbol"],
                        "timestamp": pred["timestamp"]
                    }
                    
                    # Inclure les prédictions principales en format sérialisable
                    if "prediction" in pred:
                        prediction = pred["prediction"]
                        serializable_pred["prediction"] = {}
                        
                        # Parcourir les horizons et facteurs
                        for horizon_key, horizon_data in prediction.items():
                            serializable_pred["prediction"][horizon_key] = {}
                            
                            for factor_key, factor_value in horizon_data.items():
                                # Convertir les valeurs numpy et autres en types natifs
                                if isinstance(factor_value, (np.integer, np.floating)):
                                    factor_value = float(factor_value)
                                
                                serializable_pred["prediction"][horizon_key][factor_key] = factor_value
                    
                    # Inclure les données réelles si disponibles
                    if "actual" in pred and pred["actual"] is not None:
                        serializable_pred["actual"] = {}
                        
                        for key, value in pred["actual"].items():
                            # Convertir les valeurs numpy et autres en types natifs
                            if isinstance(value, (np.integer, np.floating)):
                                value = float(value)
                            
                            serializable_pred["actual"][key] = value
                    
                    serializable_predictions.append(serializable_pred)
                
                json.dump(serializable_predictions, f, indent=2)
            
            # Sauvegarder l'historique des trades
            trades_path = os.path.join(self.data_dir, "trade_history.json")
            with open(trades_path, 'w') as f:
                json.dump(self.trade_history, f, indent=2, default=str)
            
            # Sauvegarder les performances du modèle
            performance_path = os.path.join(self.data_dir, "model_performance.json")
            with open(performance_path, 'w') as f:
                json.dump(self.model_performance, f, indent=2)
            
            # Sauvegarder l'attribution de performance
            attribution_path = os.path.join(self.data_dir, "performance_attribution.json")
            with open(attribution_path, 'w') as f:
                json.dump(self.performance_attribution, f, indent=2)
            
            logger.debug("Données de monitoring sauvegardées")
        except Exception as e:
            logger.error(f"Erreur lors de la sauvegarde des données de monitoring: {str(e)}")
    
    def _save_logs(self) -> None:
        """Sauvegarde les logs hiérarchiques"""
        try:
            logs_path = os.path.join(self.data_dir, "monitoring_logs.json")
            with open(logs_path, 'w') as f:
                json.dump(self.logs, f, indent=2)
            
            logger.debug("Logs de monitoring sauvegardés")
        except Exception as e:
            logger.error(f"Erreur lors de la sauvegarde des logs de monitoring: {str(e)}")
    
    def _load_data(self) -> None:
        """Charge les données de monitoring existantes"""
        try:
            # Charger l'historique des prédictions
            predictions_path = os.path.join(self.data_dir, "prediction_history.json")
            if os.path.exists(predictions_path):
                with open(predictions_path, 'r') as f:
                    self.prediction_history = json.load(f)
            
            # Charger l'historique des trades
            trades_path = os.path.join(self.data_dir, "trade_history.json")
            if os.path.exists(trades_path):
                with open(trades_path, 'r') as f:
                    self.trade_history = json.load(f)
            
            # Charger les performances du modèle
            performance_path = os.path.join(self.data_dir, "model_performance.json")
            if os.path.exists(performance_path):
                with open(performance_path, 'r') as f:
                    self.model_performance = json.load(f)
            
            # Charger l'attribution de performance
            attribution_path = os.path.join(self.data_dir, "performance_attribution.json")
            if os.path.exists(attribution_path):
                with open(attribution_path, 'r') as f:
                    self.performance_attribution = json.load(f)
            
            # Charger les logs
            logs_path = os.path.join(self.data_dir, "monitoring_logs.json")
            if os.path.exists(logs_path):
                with open(logs_path, 'r') as f:
                    self.logs = json.load(f)
            
            logger.info("Données de monitoring chargées")
        except Exception as e:
            logger.error(f"Erreur lors du chargement des données de monitoring: {str(e)}")
    
    def get_recent_predictions(self, symbol: str = None, horizon: str = None,
                           limit: int = 100) -> List[Dict]:
        """
        Récupère les prédictions récentes
        
        Args:
            symbol: Filtrer par paire de trading
            horizon: Filtrer par horizon de prédiction
            limit: Nombre maximum de prédictions à retourner
            
        Returns:
            Liste des prédictions récentes
        """
        # Filtrer par symbole si spécifié
        if symbol:
            filtered_predictions = [p for p in self.prediction_history if p["symbol"] == symbol]
        else:
            filtered_predictions = self.prediction_history
        
        # Filtrer par horizon si spécifié
        if horizon and filtered_predictions:
            result = []
            
            for pred in filtered_predictions:
                if "prediction" in pred and horizon in pred["prediction"]:
                    # Copier la prédiction et conserver uniquement l'horizon demandé
                    filtered_pred = pred.copy()
                    filtered_pred["prediction"] = {horizon: pred["prediction"][horizon]}
                    result.append(filtered_pred)
            
            return result[-limit:]
        
        # Retourner les prédictions filtrées
        return filtered_predictions[-limit:]
    
    def get_prediction_accuracy(self, symbol: str = None, horizon: str = None,
                             days: int = 30) -> Dict:
        """
        Calcule la précision des prédictions sur une période récente
        
        Args:
            symbol: Filtrer par paire de trading
            horizon: Filtrer par horizon de prédiction
            days: Nombre de jours à analyser
            
        Returns:
            Métriques de précision
        """
        # Calculer la date limite
        cutoff_date = (datetime.now() - timedelta(days=days)).isoformat()
        
        # Filtrer les prédictions récentes avec données réelles
        recent_predictions = [
            p for p in self.prediction_history 
            if p.get("timestamp", "") >= cutoff_date and p.get("actual") is not None
        ]
        
        # Filtrer par symbole si spécifié
        if symbol:
            recent_predictions = [p for p in recent_predictions if p["symbol"] == symbol]
        
        if not recent_predictions:
            return {
                "accuracy": 0,
                "total_predictions": 0,
                "message": "Données insuffisantes"
            }
        
        # Compteurs
        correct = 0
        total = 0
        
        # Analyser chaque prédiction
        for pred in recent_predictions:
            if "prediction" not in pred or "actual" not in pred:
                continue
            
            # Filtrer par horizon si spécifié
            horizons_to_check = [horizon] if horizon else pred["prediction"].keys()
            
            for h in horizons_to_check:
                if h in pred["prediction"]:
                    # Récupérer la direction prédite
                    predicted_direction = pred["prediction"][h].get("direction", "")
                    
                    # Récupérer la direction réelle
                    actual_direction = "HAUSSIER" if pred["actual"].get("price_change", 0) > 0 else "BAISSIER"
                    
                    # Comparer
                    if predicted_direction and predicted_direction == actual_direction:
                        correct += 1
                    
                    total += 1
        
        # Calculer la précision
        accuracy = correct / total if total > 0 else 0
        
        return {
            "accuracy": accuracy,
            "correct_predictions": correct,
            "total_predictions": total,
            "period_days": days
        }
    
    def get_trade_performance(self, symbol: str = None, days: int = 30) -> Dict:
        """
        Calcule les performances des trades sur une période récente
        
        Args:
            symbol: Filtrer par paire de trading
            days: Nombre de jours à analyser
            
        Returns:
            Métriques de performance des trades
        """
        # Calculer la date limite
        cutoff_date = (datetime.now() - timedelta(days=days)).isoformat()
        
        # Filtrer les trades récents
        recent_trades = [
            t for t in self.trade_history 
            if t.get("timestamp", "") >= cutoff_date or t.get("entry_time", "") >= cutoff_date
        ]
        
        # Filtrer par symbole si spécifié
        if symbol:
            recent_trades = [t for t in recent_trades if t.get("symbol", "") == symbol]
        
        if not recent_trades:
            return {
                "total_trades": 0,
                "win_rate": 0,
                "profit_factor": 0,
                "message": "Données insuffisantes"
            }
        
        # Calculer les métriques
        total_trades = len(recent_trades)
        winning_trades = [t for t in recent_trades if t.get("pnl_percent", 0) > 0]
        losing_trades = [t for t in recent_trades if t.get("pnl_percent", 0) <= 0]
        
        win_rate = len(winning_trades) / total_trades if total_trades > 0 else 0
        
        # Calculer le profit factor
        total_profit = sum(t.get("pnl_percent", 0) for t in winning_trades)
        total_loss = abs(sum(t.get("pnl_percent", 0) for t in losing_trades))
        
        profit_factor = total_profit / total_loss if total_loss > 0 else float('inf')
        
        # Calculer les autres métriques
        avg_profit = total_profit / len(winning_trades) if winning_trades else 0
        avg_loss = total_loss / len(losing_trades) if losing_trades else 0
        
        # Calculer le drawdown maximum
        equity_curve = self._calculate_equity_curve(recent_trades)
        max_drawdown = self._calculate_max_drawdown(equity_curve)
        
        return {
            "total_trades": total_trades,
            "winning_trades": len(winning_trades),
            "losing_trades": len(losing_trades),
            "win_rate": win_rate,
            "profit_factor": profit_factor,
            "avg_profit": avg_profit,
            "avg_loss": avg_loss,
            "max_drawdown": max_drawdown,
            "period_days": days
        }
    
    def _calculate_equity_curve(self, trades: List[Dict]) -> List[float]:
        """
        Calcule la courbe d'équité à partir des trades
        
        Args:
            trades: Liste des trades
            
        Returns:
            Courbe d'équité
        """
        # Trier les trades par date
        sorted_trades = sorted(trades, key=lambda t: t.get("entry_time", "") or t.get("timestamp", ""))
        
        # Initialiser la courbe d'équité avec un capital initial de 100
        equity = 100.0
        equity_curve = [equity]
        
        # Calculer l'équité après chaque trade
        for trade in sorted_trades:
            pnl_percent = trade.get("pnl_percent", 0)
            equity *= (1 + pnl_percent / 100)
            equity_curve.append(equity)
        
        return equity_curve
    
    def _calculate_max_drawdown(self, equity_curve: List[float]) -> float:
        """
        Calcule le drawdown maximum à partir de la courbe d'équité
        
        Args:
            equity_curve: Courbe d'équité
            
        Returns:
            Drawdown maximum en pourcentage
        """
        if not equity_curve or len(equity_curve) < 2:
            return 0.0
        
        max_dd = 0.0
        peak = equity_curve[0]
        
        for equity in equity_curve:
            if equity > peak:
                peak = equity
            
            dd = (peak - equity) / peak * 100 if peak > 0 else 0
            max_dd = max(max_dd, dd)
        
        return max_dd
    
    def generate_model_insights(self, symbol: str = None, days: int = 30) -> Dict:
        """
        Génère des insights sur les performances du modèle
        
        Args:
            symbol: Filtrer par paire de trading
            days: Nombre de jours à analyser
            
        Returns:
            Insights sur les performances du modèle
        """
        # Récupérer les prédictions récentes
        prediction_accuracy = self.get_prediction_accuracy(symbol, days=days)
        
        # Récupérer les performances des trades
        trade_performance = self.get_trade_performance(symbol, days=days)
        
        # Récupérer les métriques de performance du modèle
        model_metrics = {
            "accuracy": self.model_performance["accuracy"][-1] if self.model_performance["accuracy"] else 0,
            "precision": self.model_performance["precision"][-1] if self.model_performance["precision"] else 0,
            "recall": self.model_performance["recall"][-1] if self.model_performance["recall"] else 0,
            "f1_score": self.model_performance["f1_score"][-1] if self.model_performance["f1_score"] else 0
        }
        
        # Récupérer les attributions de performance
        attributions = {
            "model": self.performance_attribution["model_contribution"][-1] if self.performance_attribution["model_contribution"] else 0,
            "technical": self.performance_attribution["technical_contribution"][-1] if self.performance_attribution["technical_contribution"] else 0,
            "market": self.performance_attribution["market_contribution"][-1] if self.performance_attribution["market_contribution"] else 0
        }
        
        # Générer les insights
        insights = []
        
        # Insight sur la précision du modèle
        if prediction_accuracy["accuracy"] > 0.65:
            insights.append({
                "type": "strength",
                "message": f"Le modèle montre une forte précision de prédiction ({prediction_accuracy['accuracy']:.1%})"
            })
        elif prediction_accuracy["accuracy"] < 0.5:
            insights.append({
                "type": "weakness",
                "message": f"La précision de prédiction est faible ({prediction_accuracy['accuracy']:.1%})"
            })
        
        # Insight sur le win rate
        if trade_performance["win_rate"] > 0.6:
            insights.append({
                "type": "strength",
                "message": f"Excellent win rate sur les trades ({trade_performance['win_rate']:.1%})"
            })
        elif trade_performance["win_rate"] < 0.4:
            insights.append({
                "type": "weakness",
                "message": f"Win rate insuffisant ({trade_performance['win_rate']:.1%}), réévaluer la stratégie"
            })
        
        # Insight sur le profit factor
        if trade_performance["profit_factor"] > 2.0:
            insights.append({
                "type": "strength",
                "message": f"Profit factor excellent ({trade_performance['profit_factor']:.2f})"
            })
        elif trade_performance["profit_factor"] < 1.0:
            insights.append({
                "type": "weakness",
                "message": f"Profit factor inférieur à 1 ({trade_performance['profit_factor']:.2f}), les pertes dépassent les gains"
            })
        
        # Insight sur l'attribution de performance
        if attributions["model"] > 0.6:
            insights.append({
                "type": "strength",
                "message": f"Le modèle LSTM contribue fortement à la performance ({attributions['model']:.1%})"
            })
        elif attributions["technical"] > attributions["model"]:
            insights.append({
                "type": "info",
                "message": f"Les indicateurs techniques sont plus déterminants ({attributions['technical']:.1%}) que le modèle ({attributions['model']:.1%})"
            })
        
        # Calculer les tendances
        trends = self._calculate_performance_trends()
        
        return {
            "prediction_accuracy": prediction_accuracy,
            "trade_performance": trade_performance,
            "model_metrics": model_metrics,
            "performance_attribution": attributions,
            "insights": insights,
            "trends": trends
        }
    
    def _calculate_performance_trends(self) -> Dict:
        """
        Calcule les tendances de performance
        
        Returns:
            Tendances de performance
        """
        trends = {}
        
        # Calculer la tendance de précision
        if len(self.model_performance["accuracy"]) > 5:
            accuracy_trend = self.model_performance["accuracy"][-1] - self.model_performance["accuracy"][-5]
            trends["accuracy"] = {
                "direction": "improving" if accuracy_trend > 0 else "declining",
                "change": accuracy_trend
            }
        
        # Calculer la tendance de f1_score
        if len(self.model_performance["f1_score"]) > 5:
            f1_trend = self.model_performance["f1_score"][-1] - self.model_performance["f1_score"][-5]
            trends["f1_score"] = {
                "direction": "improving" if f1_trend > 0 else "declining",
                "change": f1_trend
            }
        
        # Calculer la tendance de contribution du modèle
        if len(self.performance_attribution["model_contribution"]) > 5:
            model_contrib_trend = (
                self.performance_attribution["model_contribution"][-1] - 
                self.performance_attribution["model_contribution"][-5]
            )
            trends["model_contribution"] = {
                "direction": "improving" if model_contrib_trend > 0 else "declining",
                "change": model_contrib_trend
            }
        
        return trends
    
    def create_performance_dashboard(self, days: int = 30, symbol: str = None) -> BytesIO:
        """
        Crée un tableau de bord complet des performances
        
        Args:
            days: Nombre de jours à analyser
            symbol: Filtrer par paire de trading
            
        Returns:
            Tableau de bord sous forme d'image
        """
        # Créer une figure avec plusieurs sous-graphiques
        plt.style.use('fivethirtyeight')
        fig = plt.figure(figsize=(14, 16))
        gs = gridspec.GridSpec(5, 2, figure=fig)
        
        # Récupérer les données
        insights = self.generate_model_insights(symbol, days)
        prediction_accuracy = insights["prediction_accuracy"]
        trade_performance = insights["trade_performance"]
        model_metrics = insights["model_metrics"]
        attributions = insights["performance_attribution"]
        
        # 1. Courbe d'équité
        equity_ax = fig.add_subplot(gs[0, :])
        self._plot_equity_curve(equity_ax, days, symbol)
        
        # 2. Précision des prédictions par horizon
        pred_ax = fig.add_subplot(gs[1, 0])
        self._plot_prediction_accuracy(pred_ax, days, symbol)
        
        # 3. Distribution des profits/pertes
        pnl_ax = fig.add_subplot(gs[1, 1])
        self._plot_pnl_distribution(pnl_ax, days, symbol)
        
        # 4. Performance du modèle
        model_ax = fig.add_subplot(gs[2, 0])
        self._plot_model_performance(model_ax)
        
        # 5. Attribution de performance
        attr_ax = fig.add_subplot(gs[2, 1])
        self._plot_performance_attribution(attr_ax)
        
        # 6. Répartition des trades par résultat
        trade_ax = fig.add_subplot(gs[3, 0])
        self._plot_trade_breakdown(trade_ax, days, symbol)
        
        # 7. Analyse des horizons
        horizon_ax = fig.add_subplot(gs[3, 1])
        self._plot_horizon_analysis(horizon_ax, days, symbol)
        
        # 8. Tableau des métriques clés
        metrics_ax = fig.add_subplot(gs[4, :])
        self._plot_key_metrics_table(metrics_ax, insights)
        
        # Ajuster la mise en page
        plt.tight_layout()
        
        # Sauvegarder la figure dans un buffer
        buf = BytesIO()
        plt.savefig(buf, format='png', dpi=100, bbox_inches='tight')
        buf.seek(0)
        
        # Fermer la figure pour libérer la mémoire
        plt.close(fig)
        
        return buf
    
    def _plot_equity_curve(self, ax, days: int, symbol: str = None) -> None:
        """
        Trace la courbe d'équité
        
        Args:
            ax: Axes matplotlib
            days: Nombre de jours à analyser
            symbol: Filtrer par paire de trading
        """
        # Calculer la date limite
        cutoff_date = (datetime.now() - timedelta(days=days)).isoformat()
        
        # Filtrer les trades récents
        recent_trades = [
            t for t in self.trade_history 
            if (t.get("timestamp", "") >= cutoff_date or t.get("entry_time", "") >= cutoff_date)
        ]
        
        # Filtrer par symbole si spécifié
        if symbol:
            recent_trades = [t for t in recent_trades if t.get("symbol", "") == symbol]
        
        if not recent_trades:
            ax.text(0.5, 0.5, "Données insuffisantes", ha='center', va='center')
            ax.set_title("Courbe d'équité")
            return
        
        # Trier les trades par date
        def get_timestamp(trade):
            return trade.get("exit_time", "") or trade.get("entry_time", "") or trade.get("timestamp", "")
        
        sorted_trades = sorted(recent_trades, key=get_timestamp)
        
        # Initialiser la courbe d'équité avec un capital initial de 100
        equity = 100.0
        equity_curve = [equity]
        dates = [datetime.fromisoformat(get_timestamp(sorted_trades[0]))] if sorted_trades else [datetime.now()]
        
        # Calculer l'équité après chaque trade
        for trade in sorted_trades:
            pnl_percent = trade.get("pnl_percent", 0)
            equity *= (1 + pnl_percent / 100)
            equity_curve.append(equity)
            
            # Ajouter la date
            try:
                date = datetime.fromisoformat(get_timestamp(trade))
            except:
                date = dates[-1] + timedelta(hours=1)  # Fallback
            
            dates.append(date)
        
        # Tracer la courbe d'équité
        ax.plot(dates, equity_curve, 'b-', linewidth=2)
        
        # Ajouter les points de trade
        for i, trade in enumerate(sorted_trades):
            pnl = trade.get("pnl_percent", 0)
            color = 'green' if pnl > 0 else 'red'
            marker = '^' if pnl > 0 else 'v'
            try:
                date = datetime.fromisoformat(get_timestamp(trade))
                idx = dates.index(date)
                ax.plot(date, equity_curve[idx], marker=marker, color=color, markersize=8)
            except:
                continue
        
        # Calculer le drawdown
        max_dd = self._calculate_max_drawdown(equity_curve)
        total_return = (equity_curve[-1] / equity_curve[0] - 1) * 100
        
        # Ajouter les informations sur le graphique
        ax.set_title(f"Courbe d'équité {symbol + ' ' if symbol else ''}(Rendement: {total_return:.1f}%, DD Max: {max_dd:.1f}%)")
        ax.set_ylabel("Équité (%)")
        ax.xaxis.set_major_formatter(DateFormatter("%d/%m"))
        ax.grid(True, alpha=0.3)
    
    def _plot_prediction_accuracy(self, ax, days: int, symbol: str = None) -> None:
        """
        Trace la précision des prédictions par horizon
        
        Args:
            ax: Axes matplotlib
            days: Nombre de jours à analyser
            symbol: Filtrer par paire de trading
        """
        # Horizons à analyser
        horizons = ["3h", "12h", "48h", "96h", "short_term", "medium_term", "long_term"]
        
        # Calculer la précision pour chaque horizon
        accuracies = []
        labels = []
        
        for horizon in horizons:
            accuracy = self.get_prediction_accuracy(symbol, horizon, days)
            
            # Si suffisamment de prédictions
            if accuracy["total_predictions"] > 10:
                accuracies.append(accuracy["accuracy"] * 100)  # En pourcentage
                labels.append(horizon)
        
        if not accuracies:
            ax.text(0.5, 0.5, "Données insuffisantes", ha='center', va='center')
            ax.set_title("Précision des prédictions par horizon")
            return
        
        # Tracer le graphique à barres
        colors = ['#3498db', '#2980b9', '#1f618d', '#154360', '#512E5F', '#4A235A', '#0B5345']
        bars = ax.bar(labels, accuracies, color=colors[:len(labels)])
        
        # Ajouter les valeurs au-dessus des barres
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 1,
                   f'{height:.1f}%', ha='center', va='bottom')
        
        # Ajouter une ligne horizontale à 50% (hasard)
        ax.axhline(y=50, color='r', linestyle='--', alpha=0.7)
        
        # Configurer le graphique
        ax.set_title("Précision des prédictions par horizon")
        ax.set_ylabel("Précision (%)")
        ax.set_ylim(0, 100)
        ax.grid(True, alpha=0.3, axis='y')
    
    def _plot_pnl_distribution(self, ax, days: int, symbol: str = None) -> None:
        """
        Trace la distribution des profits/pertes
        
        Args:
            ax: Axes matplotlib
            days: Nombre de jours à analyser
            symbol: Filtrer par paire de trading
        """
        # Calculer la date limite
        cutoff_date = (datetime.now() - timedelta(days=days)).isoformat()
        
        # Filtrer les trades récents
        recent_trades = [
            t for t in self.trade_history 
            if (t.get("timestamp", "") >= cutoff_date or t.get("entry_time", "") >= cutoff_date)
        ]
        
        # Filtrer par symbole si spécifié
        if symbol:
            recent_trades = [t for t in recent_trades if t.get("symbol", "") == symbol]
        
        if not recent_trades:
            ax.text(0.5, 0.5, "Données insuffisantes", ha='center', va='center')
            ax.set_title("Distribution des profits/pertes")
            return
        
        # Récupérer les pourcentages de PnL
        pnl_values = [t.get("pnl_percent", 0) for t in recent_trades]
        
        # Tracer l'histogramme
        bins = np.linspace(min(pnl_values), max(pnl_values), 20)
        ax.hist(pnl_values, bins=bins, alpha=0.7, color='skyblue', edgecolor='black')
        
        # Ajouter une ligne verticale à 0
        ax.axvline(x=0, color='r', linestyle='--', alpha=0.7)
        
        # Configurer le graphique
        ax.set_title("Distribution des profits/pertes")
        ax.set_xlabel("P&L (%)")
        ax.set_ylabel("Fréquence")
        ax.grid(True, alpha=0.3)
    
    def _plot_model_performance(self, ax) -> None:
        """
        Trace les métriques de performance du modèle
        
        Args:
            ax: Axes matplotlib
        """
        # Vérifier s'il y a suffisamment de données
        if not self.model_performance["timestamps"]:
            ax.text(0.5, 0.5, "Données insuffisantes", ha='center', va='center')
            ax.set_title("Performance du modèle")
            return
        
        # Convertir les timestamps en datetime
        try:
            dates = [datetime.fromisoformat(ts) for ts in self.model_performance["timestamps"]]
        except:
            # Fallback: créer des dates séquentielles
            dates = [datetime.now() - timedelta(days=i) for i in range(len(self.model_performance["timestamps"]), 0, -1)]
        
        # Tracer les métriques
        ax.plot(dates, self.model_performance["accuracy"], 'b-', label="Accuracy")
        ax.plot(dates, self.model_performance["f1_score"], 'g--', label="F1-Score")
        ax.plot(dates, self.model_performance["precision"], 'r-.', label="Precision")
        ax.plot(dates, self.model_performance["recall"], 'c:', label="Recall")
        
        # Configurer le graphique
        ax.set_title("Performance du modèle")
        ax.set_ylabel("Métrique")
        ax.xaxis.set_major_formatter(DateFormatter("%d/%m"))
        ax.grid(True, alpha=0.3)
        ax.legend()
    
    def _plot_performance_attribution(self, ax) -> None:
        """
        Trace l'attribution de performance
        
        Args:
            ax: Axes matplotlib
        """
        # Vérifier s'il y a suffisamment de données
        if not self.performance_attribution["timestamps"]:
            ax.text(0.5, 0.5, "Données insuffisantes", ha='center', va='center')
            ax.set_title("Attribution de performance")
            return
        
        # Convertir les timestamps en datetime
        try:
            dates = [datetime.fromisoformat(ts) for ts in self.performance_attribution["timestamps"]]
        except:
            # Fallback: créer des dates séquentielles
            dates = [datetime.now() - timedelta(days=i) for i in range(len(self.performance_attribution["timestamps"]), 0, -1)]
        
        # Tracer les attributions
        ax.stackplot(
            dates,
            self.performance_attribution["model_contribution"],
            self.performance_attribution["technical_contribution"],
            self.performance_attribution["market_contribution"],
            labels=["Modèle LSTM", "Indicateurs techniques", "Conditions de marché"],
            colors=['#3498db', '#2ecc71', '#f39c12'],
            alpha=0.7
        )
        
        # Configurer le graphique
        ax.set_title("Attribution de performance")
        ax.set_ylabel("Contribution")
        ax.xaxis.set_major_formatter(DateFormatter("%d/%m"))
        ax.grid(True, alpha=0.3)
        ax.legend(loc='upper left', fontsize='small')
    
    def _plot_trade_breakdown(self, ax, days: int, symbol: str = None) -> None:
        """
        Trace la répartition des trades par résultat
        
        Args:
            ax: Axes matplotlib
            days: Nombre de jours à analyser
            symbol: Filtrer par paire de trading
        """
        # Calculer la date limite
        cutoff_date = (datetime.now() - timedelta(days=days)).isoformat()
        
        # Filtrer les trades récents
        recent_trades = [
            t for t in self.trade_history 
            if (t.get("timestamp", "") >= cutoff_date or t.get("entry_time", "") >= cutoff_date)
        ]
        
        # Filtrer par symbole si spécifié
        if symbol:
            recent_trades = [t for t in recent_trades if t.get("symbol", "") == symbol]
        
        if not recent_trades:
            ax.text(0.5, 0.5, "Données insuffisantes", ha='center', va='center')
            ax.set_title("Répartition des trades")
            return
        
        # Catégoriser les trades
        categories = {
            "profit_major": len([t for t in recent_trades if t.get("pnl_percent", 0) > 5]),
            "profit_minor": len([t for t in recent_trades if 0 < t.get("pnl_percent", 0) <= 5]),
            "loss_minor": len([t for t in recent_trades if -5 <= t.get("pnl_percent", 0) < 0]),
            "loss_major": len([t for t in recent_trades if t.get("pnl_percent", 0) < -5])
        }
        
        # Tracer le graphique à camembert
        labels = ["Profit majeur (>5%)", "Profit mineur (0-5%)", "Perte mineure (0-5%)", "Perte majeure (>5%)"]
        sizes = [categories["profit_major"], categories["profit_minor"], categories["loss_minor"], categories["loss_major"]]
        colors = ['#27ae60', '#2ecc71', '#e74c3c', '#c0392b']
        explode = (0.1, 0, 0, 0.1)  # Faire ressortir les catégories extrêmes
        
        ax.pie(sizes, explode=explode, labels=labels, colors=colors, autopct='%1.1f%%',
              shadow=True, startangle=90)
        ax.axis('equal')  # Pour avoir un cercle parfait
        
        # Configurer le graphique
        ax.set_title("Répartition des trades par résultat")
    
    def _plot_horizon_analysis(self, ax, days: int, symbol: str = None) -> None:
        """
        Trace l'analyse des horizons
        
        Args:
            ax: Axes matplotlib
            days: Nombre de jours à analyser
            symbol: Filtrer par paire de trading
        """
        # Horizons à analyser
        horizons = ["3h", "12h", "48h", "96h", "short_term", "medium_term", "long_term"]
        
        # Calculer la précision pour chaque horizon
        accuracies = {}
        
        for horizon in horizons:
            accuracy = self.get_prediction_accuracy(symbol, horizon, days)
            
            # Si suffisamment de prédictions
            if accuracy["total_predictions"] > 10:
                accuracies[horizon] = {
                    "accuracy": accuracy["accuracy"],
                    "total": accuracy["total_predictions"]
                }
        
        if not accuracies:
            ax.text(0.5, 0.5, "Données insuffisantes", ha='center', va='center')
            ax.set_title("Analyse des horizons")
            return
        
        # Créer un dataframe pour l'analyse
        df = pd.DataFrame({
            "Horizon": list(accuracies.keys()),
            "Précision": [acc["accuracy"] * 100 for acc in accuracies.values()],
            "Nombre de prédictions": [acc["total"] for acc in accuracies.values()]
        })
        
        # Trier par précision
        df = df.sort_values("Précision", ascending=False)
        
        # Tracer le graphique à barres
        sns.barplot(x="Horizon", y="Précision", data=df, ax=ax, palette="viridis")
        
        # Ajouter une ligne horizontale à 50% (hasard)
        ax.axhline(y=50, color='r', linestyle='--', alpha=0.7)
        
        # Ajouter les nombres de prédictions au-dessus des barres
        for i, v in enumerate(df["Nombre de prédictions"]):
            ax.text(i, df["Précision"].iloc[i] + 1, str(v), ha='center')
        
        # Configurer le graphique
        ax.set_title("Analyse des horizons (précision)")
        ax.set_ylim(0, 100)
        ax.grid(True, alpha=0.3, axis='y')
    
    def _plot_key_metrics_table(self, ax, insights: Dict) -> None:
        """
        Trace un tableau des métriques clés
        
        Args:
            ax: Axes matplotlib
            insights: Insights sur les performances du modèle
        """
        # Désactiver les axes
        ax.axis('off')
        
        # Extraire les données clés
        prediction_accuracy = insights["prediction_accuracy"]
        trade_performance = insights["trade_performance"]
        model_metrics = insights["model_metrics"]
        attributions = insights["performance_attribution"]
        trends = insights.get("trends", {})
        
        # Créer les données du tableau
        data = [
            ["Métrique", "Valeur", "Tendance"],
            ["Précision de prédiction", f"{prediction_accuracy['accuracy']:.1%}", self._get_trend_arrow(trends.get("accuracy", {}))],
            ["Win rate", f"{trade_performance['win_rate']:.1%}", ""],
            ["Profit factor", f"{trade_performance['profit_factor']:.2f}", ""],
            ["Drawdown maximum", f"{trade_performance['max_drawdown']:.1f}%", ""],
            ["F1-Score du modèle", f"{model_metrics['f1_score']:.2f}", self._get_trend_arrow(trends.get("f1_score", {}))],
            ["Contribution du modèle", f"{attributions['model']:.1%}", self._get_trend_arrow(trends.get("model_contribution", {}))]
        ]
        
        # Créer le tableau
        table = ax.table(
            cellText=data,
            cellLoc='center',
            loc='center',
            colWidths=[0.4, 0.3, 0.3]
        )
        
        # Styliser le tableau
        table.auto_set_font_size(False)
        table.set_fontsize(12)
        table.scale(1, 1.5)
        
        # Styliser l'en-tête
        for i in range(3):
            table[(0, i)].set_facecolor('#3498db')
            table[(0, i)].set_text_props(color='white', fontweight='bold')
        
        # Styliser les lignes alternées
        for i in range(1, len(data)):
            if i % 2 == 0:
                for j in range(3):
                    table[(i, j)].set_facecolor('#f5f5f5')
    
    def _get_trend_arrow(self, trend: Dict) -> str:
        """
        Retourne une flèche indiquant la tendance
        
        Args:
            trend: Dictionnaire de tendance
            
        Returns:
            Flèche de tendance
        """
        direction = trend.get("direction", "")
        
        if direction == "improving":
            return "↑"
        elif direction == "declining":
            return "↓"
        else:
            return "→"
    
    def create_plotly_dashboard(self, days: int = 30, symbol: str = None) -> Dict:
        """
        Crée un tableau de bord interactif avec Plotly
        
        Args:
            days: Nombre de jours à analyser
            symbol: Filtrer par paire de trading
            
        Returns:
            Dictionnaire avec les figures Plotly
        """
        # Récupérer les données
        insights = self.generate_model_insights(symbol, days)
        
        # 1. Courbe d'équité
        equity_fig = self._create_plotly_equity_curve(days, symbol)
        
        # 2. Précision des prédictions par horizon
        prediction_fig = self._create_plotly_prediction_accuracy(days, symbol)
        
        # 3. Distribution des profits/pertes
        pnl_fig = self._create_plotly_pnl_distribution(days, symbol)
        
        # 4. Performance du modèle
        model_fig = self._create_plotly_model_performance()
        
        # 5. Attribution de performance
        attribution_fig = self._create_plotly_performance_attribution()
        
        # 6. Tableau des métriques clés
        metrics_fig = self._create_plotly_metrics_table(insights)
        
        return {
            "equity_curve": equity_fig,
            "prediction_accuracy": prediction_fig,
            "pnl_distribution": pnl_fig,
            "model_performance": model_fig,
            "performance_attribution": attribution_fig,
            "metrics_table": metrics_fig,
            "insights": insights
        }
    
    def _create_plotly_equity_curve(self, days: int, symbol: str = None) -> go.Figure:
        """
        Crée une courbe d'équité interactive avec Plotly
        
        Args:
            days: Nombre de jours à analyser
            symbol: Filtrer par paire de trading
            
        Returns:
            Figure Plotly
        """
        # Calculer la date limite
        cutoff_date = (datetime.now() - timedelta(days=days)).isoformat()
        
        # Filtrer les trades récents
        recent_trades = [
            t for t in self.trade_history 
            if (t.get("timestamp", "") >= cutoff_date or t.get("entry_time", "") >= cutoff_date)
        ]
        
        # Filtrer par symbole si spécifié
        if symbol:
            recent_trades = [t for t in recent_trades if t.get("symbol", "") == symbol]
        
        if not recent_trades:
            fig = go.Figure()
            fig.add_annotation(
                text="Données insuffisantes",
                xref="paper", yref="paper",
                x=0.5, y=0.5,
                showarrow=False,
                font=dict(size=20)
            )
            fig.update_layout(title="Courbe d'équité")
            return fig
        
        # Trier les trades par date
        def get_timestamp(trade):
            return trade.get("exit_time", "") or trade.get("entry_time", "") or trade.get("timestamp", "")
        
        sorted_trades = sorted(recent_trades, key=get_timestamp)
        
        # Initialiser la courbe d'équité avec un capital initial de 100
        equity = 100.0
        equity_curve = [equity]
        dates = [datetime.fromisoformat(get_timestamp(sorted_trades[0]))] if sorted_trades else [datetime.now()]
        trade_pnl = []
        
        # Calculer l'équité après chaque trade
        for trade in sorted_trades:
            pnl_percent = trade.get("pnl_percent", 0)
            equity *= (1 + pnl_percent / 100)
            equity_curve.append(equity)
            trade_pnl.append(pnl_percent)
            
            # Ajouter la date
            try:
                date = datetime.fromisoformat(get_timestamp(trade))
            except:
                date = dates[-1] + timedelta(hours=1)  # Fallback
            
            dates.append(date)
        
        # Créer la figure
        fig = go.Figure()
        
        # Ajouter la courbe d'équité
        fig.add_trace(go.Scatter(
            x=dates,
            y=equity_curve,
            mode='lines',
            name='Équité',
            line=dict(color='blue', width=2)
        ))
        
        # Ajouter les points de trade
        winning_trades_x = []
        winning_trades_y = []
        losing_trades_x = []
        losing_trades_y = []
        
        for i, trade in enumerate(sorted_trades):
            pnl = trade.get("pnl_percent", 0)
            
            try:
                date = datetime.fromisoformat(get_timestamp(trade))
                idx = dates.index(date)
                
                if pnl > 0:
                    winning_trades_x.append(date)
                    winning_trades_y.append(equity_curve[idx])
                else:
                    losing_trades_x.append(date)
                    losing_trades_y.append(equity_curve[idx])
            except:
                continue
        
        # Ajouter les trades gagnants
        fig.add_trace(go.Scatter(
            x=winning_trades_x,
            y=winning_trades_y,
            mode='markers',
            name='Trades gagnants',
            marker=dict(
                color='green',
                size=10,
                symbol='triangle-up'
            )
        ))
        
        # Ajouter les trades perdants
        fig.add_trace(go.Scatter(
            x=losing_trades_x,
            y=losing_trades_y,
            mode='markers',
            name='Trades perdants',
            marker=dict(
                color='red',
                size=10,
                symbol='triangle-down'
            )
        ))
        
        # Calculer le drawdown
        max_dd = self._calculate_max_drawdown(equity_curve)
        total_return = (equity_curve[-1] / equity_curve[0] - 1) * 100
        
        # Configurer la mise en page
        title = f"Courbe d'équité {symbol + ' ' if symbol else ''}"
        title += f"(Rendement: {total_return:.1f}%, DD Max: {max_dd:.1f}%)"
        
        fig.update_layout(
            title=title,
            xaxis_title="Date",
            yaxis_title="Équité (%)",
            legend=dict(
                x=0.01,
                y=0.99,
                bordercolor="Black",
                borderwidth=1
            ),
            hovermode="x unified",
            plot_bgcolor='white',
            margin=dict(l=20, r=20, t=50, b=20)
        )
        
        # Ajouter une grille
        fig.update_xaxes(
            showgrid=True,
            gridcolor='lightgray'
        )
        fig.update_yaxes(
            showgrid=True,
            gridcolor='lightgray'
        )
        
        return fig
    
    def _create_plotly_prediction_accuracy(self, days: int, symbol: str = None) -> go.Figure:
        """
        Crée un graphique de précision des prédictions interactif avec Plotly
        
        Args:
            days: Nombre de jours à analyser
            symbol: Filtrer par paire de trading
            
        Returns:
            Figure Plotly
        """
        # Horizons à analyser
        horizons = ["3h", "12h", "48h", "96h", "short_term", "medium_term", "long_term"]
        
        # Calculer la précision pour chaque horizon
        accuracies = []
        labels = []
        counts = []
        
        for horizon in horizons:
            accuracy = self.get_prediction_accuracy(symbol, horizon, days)
            
            # Si suffisamment de prédictions
            if accuracy["total_predictions"] > 10:
                accuracies.append(accuracy["accuracy"] * 100)  # En pourcentage
                labels.append(horizon)
                counts.append(accuracy["total_predictions"])
        
        if not accuracies:
            fig = go.Figure()
            fig.add_annotation(
                text="Données insuffisantes",
                xref="paper", yref="paper",
                x=0.5, y=0.5,
                showarrow=False,
                font=dict(size=20)
            )
            fig.update_layout(title="Précision des prédictions par horizon")
            return fig
        
        # Créer la figure
        fig = go.Figure()
        
        # Ajouter le graphique à barres
        fig.add_trace(go.Bar(
            x=labels,
            y=accuracies,
            text=[f"{acc:.1f}%<br>({count})" for acc, count in zip(accuracies, counts)],
            textposition='outside',
            marker_color=['#3498db', '#2980b9', '#1f618d', '#154360', '#512E5F', '#4A235A', '#0B5345'],
            hovertemplate="Horizon: %{x}<br>Précision: %{y:.1f}%<br>Échantillons: %{text}<extra></extra>"
        ))
        
        # Ajouter une ligne horizontale à 50% (hasard)
        fig.add_shape(
            type="line",
            x0=-0.5,
            y0=50,
            x1=len(labels) - 0.5,
            y1=50,
            line=dict(
                color="red",
                width=2,
                dash="dash",
            )
        )
        
        # Configurer la mise en page
        fig.update_layout(
            title="Précision des prédictions par horizon",
            xaxis_title="Horizon",
            yaxis_title="Précision (%)",
            yaxis_range=[0, 100],
            plot_bgcolor='white',
            margin=dict(l=20, r=20, t=50, b=20)
        )
        
        # Ajouter une grille
        fig.update_xaxes(
            showgrid=False
        )
        fig.update_yaxes(
            showgrid=True,
            gridcolor='lightgray'
        )
        
        return fig
    
    def _create_plotly_pnl_distribution(self, days: int, symbol: str = None) -> go.Figure:
        """
        Crée un histogramme interactif de distribution des P&L avec Plotly
        
        Args:
            days: Nombre de jours à analyser
            symbol: Filtrer par paire de trading
            
        Returns:
            Figure Plotly
        """
        # Calculer la date limite
        cutoff_date = (datetime.now() - timedelta(days=days)).isoformat()
        
        # Filtrer les trades récents
        recent_trades = [
            t for t in self.trade_history 
            if (t.get("timestamp", "") >= cutoff_date or t.get("entry_time", "") >= cutoff_date)
        ]
        
        # Filtrer par symbole si spécifié
        if symbol:
            recent_trades = [t for t in recent_trades if t.get("symbol", "") == symbol]
        
        if not recent_trades:
            fig = go.Figure()
            fig.add_annotation(
                text="Données insuffisantes",
                xref="paper", yref="paper",
                x=0.5, y=0.5,
                showarrow=False,
                font=dict(size=20)
            )
            fig.update_layout(title="Distribution des profits/pertes")
            return fig
        
        # Récupérer les pourcentages de PnL
        pnl_values = [t.get("pnl_percent", 0) for t in recent_trades]
        
        # Créer l'histogramme
        fig = go.Figure()
        
        fig.add_trace(go.Histogram(
            x=pnl_values,
            histnorm='',
            marker=dict(
                color='skyblue',
                line=dict(
                    color='darkblue',
                    width=1
                )
            ),
            hovertemplate="P&L: %{x:.2f}%<br>Fréquence: %{y}<extra></extra>"
        ))
        
        # Ajouter une ligne verticale à 0
        fig.add_shape(
            type="line",
            x0=0,
            y0=0,
            x1=0,
            y1=1,
            yref="paper",
            line=dict(
                color="red",
                width=2,
                dash="dash",
            )
        )
        
        # Configurer la mise en page
        fig.update_layout(
            title="Distribution des profits/pertes",
            xaxis_title="P&L (%)",
            yaxis_title="Fréquence",
            bargap=0.05,
            plot_bgcolor='white',
            margin=dict(l=20, r=20, t=50, b=20)
        )
        
        # Ajouter une grille
        fig.update_xaxes(
            showgrid=True,
            gridcolor='lightgray'
        )
        fig.update_yaxes(
            showgrid=True,
            gridcolor='lightgray'
        )
        
        return fig
    
    def _create_plotly_model_performance(self) -> go.Figure:
        """
        Crée un graphique interactif des métriques de performance du modèle avec Plotly
        
        Returns:
            Figure Plotly
        """
        # Vérifier s'il y a suffisamment de données
        if not self.model_performance["timestamps"]:
            fig = go.Figure()
            fig.add_annotation(
                text="Données insuffisantes",
                xref="paper", yref="paper",
                x=0.5, y=0.5,
                showarrow=False,
                font=dict(size=20)
            )
            fig.update_layout(title="Performance du modèle")
            return fig
        
        # Convertir les timestamps en datetime
        try:
            dates = [datetime.fromisoformat(ts) for ts in self.model_performance["timestamps"]]
        except:
            # Fallback: créer des dates séquentielles
            dates = [datetime.now() - timedelta(days=i) for i in range(len(self.model_performance["timestamps"]), 0, -1)]
        
        # Créer la figure
        fig = go.Figure()
        
        # Ajouter les métriques
        fig.add_trace(go.Scatter(
            x=dates,
            y=self.model_performance["accuracy"],
            mode='lines+markers',
            name='Accuracy',
            line=dict(color='blue', width=2)
        ))
        
        fig.add_trace(go.Scatter(
            x=dates,
            y=self.model_performance["f1_score"],
            mode='lines+markers',
            name='F1-Score',
            line=dict(color='green', width=2, dash='dash')
        ))
        
        fig.add_trace(go.Scatter(
            x=dates,
            y=self.model_performance["precision"],
            mode='lines+markers',
            name='Precision',
            line=dict(color='red', width=2, dash='dot')
        ))
        
        fig.add_trace(go.Scatter(
            x=dates,
            y=self.model_performance["recall"],
            mode='lines+markers',
            name='Recall',
            line=dict(color='purple', width=2, dash='dashdot')
        ))
        
        # Configurer la mise en page
        fig.update_layout(
            title="Performance du modèle",
            xaxis_title="Date",
            yaxis_title="Métrique",
            legend=dict(
                x=0.01,
                y=0.99,
                bordercolor="Black",
                borderwidth=1
            ),
            hovermode="x unified",
            plot_bgcolor='white',
            margin=dict(l=20, r=20, t=50, b=20)
        )
        
        # Ajouter une grille
        fig.update_xaxes(
            showgrid=True,
            gridcolor='lightgray'
        )
        fig.update_yaxes(
            showgrid=True,
            gridcolor='lightgray'
        )
        
        return fig
    
    def _create_plotly_performance_attribution(self) -> go.Figure:
        """
        Crée un graphique interactif d'attribution de performance avec Plotly
        
        Returns:
            Figure Plotly
        """
        # Vérifier s'il y a suffisamment de données
        if not self.performance_attribution["timestamps"]:
            fig = go.Figure()
            fig.add_annotation(
                text="Données insuffisantes",
                xref="paper", yref="paper",
                x=0.5, y=0.5,
                showarrow=False,
                font=dict(size=20)
            )
            fig.update_layout(title="Attribution de performance")
            return fig
        
        # Convertir les timestamps en datetime
        try:
            dates = [datetime.fromisoformat(ts) for ts in self.performance_attribution["timestamps"]]
        except:
            # Fallback: créer des dates séquentielles
            dates = [datetime.now() - timedelta(days=i) for i in range(len(self.performance_attribution["timestamps"]), 0, -1)]
        
        # Créer la figure
        fig = go.Figure()
        
        # Ajouter les traces d'aire empilée
        fig.add_trace(go.Scatter(
            x=dates,
            y=self.performance_attribution["model_contribution"],
            mode='lines',
            name='Modèle LSTM',
            stackgroup='one',
            fillcolor='#3498db',
            line=dict(width=0)
        ))
        
        fig.add_trace(go.Scatter(
            x=dates,
            y=self.performance_attribution["technical_contribution"],
            mode='lines',
            name='Indicateurs techniques',
            stackgroup='one',
            fillcolor='#2ecc71',
            line=dict(width=0)
        ))
        
        fig.add_trace(go.Scatter(
            x=dates,
            y=self.performance_attribution["market_contribution"],
            mode='lines',
            name='Conditions de marché',
            stackgroup='one',
            fillcolor='#f39c12',
            line=dict(width=0)
        ))
        
        # Configurer la mise en page
        fig.update_layout(
            title="Attribution de performance",
            xaxis_title="Date",
            yaxis_title="Contribution",
            legend=dict(
                x=0.01,
                y=0.99,
                bordercolor="Black",
                borderwidth=1
            ),
            hovermode="x unified",
            plot_bgcolor='white',
            margin=dict(l=20, r=20, t=50, b=20)
        )
        
        # Ajouter une grille
        fig.update_xaxes(
            showgrid=True,
            gridcolor='lightgray'
        )
        fig.update_yaxes(
            showgrid=True,
            gridcolor='lightgray'
        )
        
        return fig
    
    def _create_plotly_metrics_table(self, insights: Dict) -> go.Figure:
        """
        Crée un tableau des métriques clés avec Plotly
        
        Args:
            insights: Insights sur les performances du modèle
            
        Returns:
            Figure Plotly
        """
        # Extraire les données clés
        prediction_accuracy = insights["prediction_accuracy"]
        trade_performance = insights["trade_performance"]
        model_metrics = insights["model_metrics"]
        attributions = insights["performance_attribution"]
        trends = insights.get("trends", {})
        
        # Créer les données du tableau
        headers = ["Métrique", "Valeur", "Tendance"]
        cells = [
            ["Précision de prédiction", f"{prediction_accuracy['accuracy']:.1%}", self._get_trend_arrow(trends.get("accuracy", {}))],
            ["Win rate", f"{trade_performance['win_rate']:.1%}", ""],
            ["Profit factor", f"{trade_performance['profit_factor']:.2f}", ""],
            ["Drawdown maximum", f"{trade_performance['max_drawdown']:.1f}%", ""],
            ["F1-Score du modèle", f"{model_metrics['f1_score']:.2f}", self._get_trend_arrow(trends.get("f1_score", {}))],
            ["Contribution du modèle", f"{attributions['model']:.1%}", self._get_trend_arrow(trends.get("model_contribution", {}))]
        ]
        
        # Transposer pour le format Plotly
        cell_values = [headers]
        for row in cells:
            cell_values.append(row)
        
        # Couleurs pour les cellules
        colors = [[None, None, None]]  # En-tête
        for i, row in enumerate(cells):
            if i % 2 == 0:
                colors.append(['white', 'white', 'white'])
            else:
                colors.append(['#f5f5f5', '#f5f5f5', '#f5f5f5'])
        
        # Créer la figure
        fig = go.Figure(data=[go.Table(
            header=dict(
                values=headers,
                align=['left', 'center', 'center'],
                fill_color='#3498db',
                font=dict(color='white', size=14)
            ),
            cells=dict(
                values=[
                    [cell[0] for cell in cells],
                    [cell[1] for cell in cells],
                    [cell[2] for cell in cells]
                ],
                align=['left', 'center', 'center'],
                fill_color=[color[0] for color in colors[1:]],
                font=dict(size=12)
            )
        )])
        
        # Configurer la mise en page
        fig.update_layout(
            title="Métriques clés",
            margin=dict(l=20, r=20, t=50, b=20)
        )
        
        return fig

def test_monitor():
    """
    Fonction de test pour le monitor
    """
    monitor = ModelMonitor()
    
    # Ajouter des prédictions fictives
    for i in range(100):
        date = datetime.now() - timedelta(days=i)
        
        # Créer une prédiction fictive
        prediction = {
            "3h": {
                "direction": "HAUSSIER" if np.random.random() > 0.5 else "BAISSIER",
                "direction_probability": np.random.uniform(50, 100),
                "predicted_volatility": np.random.uniform(0.01, 0.05),
                "predicted_volume": np.random.uniform(0.8, 1.2),
                "predicted_momentum": np.random.uniform(-0.5, 0.5),
                "confidence": np.random.uniform(0.3, 0.9)
            },
            "12h": {
                "direction": "HAUSSIER" if np.random.random() > 0.5 else "BAISSIER",
                "direction_probability": np.random.uniform(50, 100),
                "predicted_volatility": np.random.uniform(0.01, 0.05),
                "predicted_volume": np.random.uniform(0.8, 1.2),
                "predicted_momentum": np.random.uniform(-0.5, 0.5),
                "confidence": np.random.uniform(0.3, 0.9)
            }
        }
        
        # Créer des données réelles fictives
        actual_data = {
            "price_change": np.random.uniform(-3, 3)
        }
        
        # Enregistrer la prédiction
        monitor.record_prediction("BTCUSDT", prediction, actual_data, date.isoformat())
    
    # Ajouter des trades fictifs
    for i in range(50):
        date = datetime.now() - timedelta(days=i)
        
        # Créer un trade fictif
        trade = {
            "symbol": "BTCUSDT",
            "entry_time": (date - timedelta(hours=6)).isoformat(),
            "exit_time": date.isoformat(),
            "entry_price": 20000 + np.random.uniform(-1000, 1000),
            "exit_price": 20000 + np.random.uniform(-1000, 1000),
            "pnl_percent": np.random.uniform(-10, 15),
            "pnl_absolute": np.random.uniform(-20, 30),
            "side": "BUY" if np.random.random() > 0.5 else "SELL",
            "leverage": 1
        }
        
        # Enregistrer le trade
        monitor.record_trade(trade)
    
    # Ajouter des métriques de performance fictives
    for i in range(10):
        date = datetime.now() - timedelta(days=i*3)
        
        metrics = {
            "accuracy": np.random.uniform(0.55, 0.75),
            "precision": np.random.uniform(0.5, 0.8),
            "recall": np.random.uniform(0.5, 0.8),
            "f1_score": np.random.uniform(0.55, 0.75)
        }
        
        monitor.update_model_performance(metrics, date.isoformat())
    
    # Ajouter des attributions de performance fictives
    for i in range(10):
        date = datetime.now() - timedelta(days=i*3)
        
        # Générer des contributions aléatoires qui somment à 1
        model_contrib = np.random.uniform(0.3, 0.6)
        technical_contrib = np.random.uniform(0.2, 0.4)
        market_contrib = 1 - model_contrib - technical_contrib
        
        monitor.update_performance_attribution(
            model_contrib,
            technical_contrib,
            market_contrib,
            date.isoformat()
        )
    
    # Générer et sauvegarder un tableau de bord
    dashboard_image = monitor.create_performance_dashboard(days=30)
    
    with open("dashboard.png", "wb") as f:
        f.write(dashboard_image.getvalue())
    
    print("Tableau de bord sauvegardé dans dashboard.png")
    
    return monitor

if __name__ == "__main__":
    test_monitor()