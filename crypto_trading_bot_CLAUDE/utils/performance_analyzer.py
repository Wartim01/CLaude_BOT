"""
Module d'analyse de performances pour évaluer l'efficacité du système de trading
et générer des visualisations et des rapports d'analyse
"""
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
from typing import Dict, List, Optional, Union, Any, Tuple
from datetime import datetime, timedelta
import json
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from config.config import DATA_DIR
from utils.logger import setup_logger

logger = setup_logger("performance_analyzer")

class PerformanceAnalyzer:
    """
    Analyseur de performances pour évaluer l'efficacité du système de trading
    et produire des visualisations et rapports détaillés
    """
    def __init__(self, output_dir: Optional[str] = None):
        """
        Initialise l'analyseur de performances
        
        Args:
            output_dir: Répertoire de sortie pour les rapports et visualisations
        """
        self.output_dir = output_dir or os.path.join(DATA_DIR, "performance_reports")
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Styles pour les visualisations
        sns.set_style("whitegrid")
        plt.rcParams.update({
            'font.size': 10,
            'axes.titlesize': 14,
            'axes.labelsize': 12,
            'xtick.labelsize': 10,
            'ytick.labelsize': 10,
            'legend.fontsize': 10,
            'figure.titlesize': 16
        })
    
    def analyze_decision_engine(self, decision_history: Dict, 
                             performance_metrics: Dict) -> Dict:
        """
        Analyse les performances du moteur de décision
        
        Args:
            decision_history: Historique des décisions par symbole
            performance_metrics: Métriques de performance globales
            
        Returns:
            Résultats de l'analyse
        """
        if not decision_history:
            return {
                "success": False,
                "error": "Historique des décisions vide"
            }
        
        # 1. Analyser les décisions complétées (avec un résultat)
        completed_decisions = []
        for symbol, decisions in decision_history.items():
            for decision in decisions:
                if decision["result"]["actual_outcome"] is not None:
                    # Ajouter le symbole à la décision pour faciliter l'analyse
                    decision["symbol"] = symbol
                    completed_decisions.append(decision)
        
        if not completed_decisions:
            return {
                "success": False,
                "error": "Aucune décision complétée trouvée"
            }
        
        # 2. Convertir en DataFrame pour faciliter l'analyse
        decisions_df = pd.DataFrame([
            {
                "symbol": d["symbol"],
                "timestamp": d["timestamp"],
                "direction": d["decision"]["direction"],
                "signal_type": d["decision"].get("signal_type", "UNKNOWN"),
                "confidence": d["decision"].get("confidence", 0.0),
                "weighted_score": d["decision"].get("weighted_score", 50.0),
                "actual_outcome": d["result"]["actual_outcome"],
                "pnl": d["result"].get("pnl", 0.0),
                "success": d["result"].get("success", False),
                "strategy_weight": d["decision"].get("contributors", {}).get("strategy", {}).get("weight", 0.0),
                "risk_weight": d["decision"].get("contributors", {}).get("risk", {}).get("weight", 0.0),
                "sentiment_weight": d["decision"].get("contributors", {}).get("sentiment", {}).get("weight", 0.0),
                "technical_weight": d["decision"].get("contributors", {}).get("technical", {}).get("weight", 0.0)
            }
            for d in completed_decisions
        ])
        
        # Convertir timestamp en datetime
        decisions_df["timestamp"] = pd.to_datetime(decisions_df["timestamp"])
        decisions_df = decisions_df.sort_values("timestamp")
        
        # 3. Analyser les performances par direction
        direction_analysis = decisions_df.groupby("direction").agg(
            count=("symbol", "size"),
            success_count=("success", lambda x: sum(x)),
            failure_count=("success", lambda x: sum(~x)),
            win_rate=("success", lambda x: sum(x) / len(x) * 100 if len(x) > 0 else 0),
            avg_pnl=("pnl", "mean"),
            total_pnl=("pnl", "sum"),
            avg_confidence=("confidence", "mean")
        ).reset_index()
        
        # 4. Analyser les performances par symbole
        symbol_analysis = decisions_df.groupby("symbol").agg(
            count=("symbol", "size"),
            success_count=("success", lambda x: sum(x)),
            failure_count=("success", lambda x: sum(~x)),
            win_rate=("success", lambda x: sum(x) / len(x) * 100 if len(x) > 0 else 0),
            avg_pnl=("pnl", "mean"),
            total_pnl=("pnl", "sum")
        ).reset_index()
        
        # 5. Analyser les performances par niveau de confiance
        # Créer des tranches de confiance
        decisions_df["confidence_range"] = pd.cut(
            decisions_df["confidence"],
            bins=[0, 0.25, 0.5, 0.75, 1.0],
            labels=["0-25%", "25-50%", "50-75%", "75-100%"]
        )
        
        confidence_analysis = decisions_df.groupby("confidence_range").agg(
            count=("symbol", "size"),
            success_count=("success", lambda x: sum(x)),
            win_rate=("success", lambda x: sum(x) / len(x) * 100 if len(x) > 0 else 0),
            avg_pnl=("pnl", "mean")
        ).reset_index()
        
        # 6. Analyser l'évolution des performances dans le temps
        # Grouper par semaine
        decisions_df["week"] = decisions_df["timestamp"].dt.to_period("W")
        time_analysis = decisions_df.groupby("week").agg(
            count=("symbol", "size"),
            win_rate=("success", lambda x: sum(x) / len(x) * 100 if len(x) > 0 else 0),
            avg_pnl=("pnl", "mean"),
            cumulative_pnl=("pnl", "sum")
        ).reset_index()
        
        # Convertir la période en datetime pour le plotting
        time_analysis["week_start"] = time_analysis["week"].dt.start_time
        
        # Calculer le PnL cumulatif
        time_analysis["cumulative_pnl"] = time_analysis["cumulative_pnl"].cumsum()
        
        # 7. Générer les visualisations
        visualization_paths = self._generate_decision_visualizations(
            decisions_df, 
            direction_analysis, 
            symbol_analysis, 
            confidence_analysis, 
            time_analysis
        )
        
        # 8. Analyser l'importance des facteurs (poids)
        weight_columns = ["strategy_weight", "risk_weight", "sentiment_weight", "technical_weight"]
        avg_weights = decisions_df[weight_columns].mean().to_dict()
        
        # Analyser les poids par résultat
        successful_decisions = decisions_df[decisions_df["success"] == True]
        failed_decisions = decisions_df[decisions_df["success"] == False]
        
        weights_by_outcome = {
            "successful": successful_decisions[weight_columns].mean().to_dict() if not successful_decisions.empty else None,
            "failed": failed_decisions[weight_columns].mean().to_dict() if not failed_decisions.empty else None
        }
        
        # 9. Calculer les métriques supplémentaires
        metrics = {
            "total_decisions": len(decisions_df),
            "successful_decisions": sum(decisions_df["success"]),
            "win_rate": sum(decisions_df["success"]) / len(decisions_df) * 100 if len(decisions_df) > 0 else 0,
            "avg_pnl": decisions_df["pnl"].mean(),
            "total_pnl": decisions_df["pnl"].sum(),
            "max_win": decisions_df["pnl"].max(),
            "max_loss": decisions_df["pnl"].min(),
            "profit_factor": (decisions_df[decisions_df["pnl"] > 0]["pnl"].sum() / 
                            abs(decisions_df[decisions_df["pnl"] < 0]["pnl"].sum())
                            if decisions_df[decisions_df["pnl"] < 0]["pnl"].sum() != 0 else float('inf')),
            "sharpe_ratio": (decisions_df["pnl"].mean() / decisions_df["pnl"].std()
                           if decisions_df["pnl"].std() != 0 else 0)
        }
        
        # 10. Créer le rapport final
        report = {
            "success": True,
            "analysis_timestamp": datetime.now().isoformat(),
            "total_decisions_analyzed": len(decisions_df),
            "metrics": metrics,
            "direction_performance": direction_analysis.to_dict(orient="records"),
            "symbol_performance": symbol_analysis.to_dict(orient="records"),
            "confidence_performance": confidence_analysis.to_dict(orient="records"),
            "time_analysis": {
                "periods": time_analysis["week_start"].dt.strftime("%Y-%m-%d").tolist(),
                "win_rates": time_analysis["win_rate"].tolist(),
                "cumulative_pnl": time_analysis["cumulative_pnl"].tolist()
            },
            "average_weights": avg_weights,
            "weights_by_outcome": weights_by_outcome,
            "visualization_paths": visualization_paths
        }
        
        # Sauvegarder le rapport
        report_path = os.path.join(self.output_dir, f"decision_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
        try:
            with open(report_path, 'w') as f:
                json.dump(report, f, indent=2, default=str)
            report["report_path"] = report_path
        except Exception as e:
            logger.error(f"Erreur lors de la sauvegarde du rapport: {str(e)}")
        
        return report
    
    def _generate_decision_visualizations(self, decisions_df: pd.DataFrame,
                                        direction_analysis: pd.DataFrame,
                                        symbol_analysis: pd.DataFrame,
                                        confidence_analysis: pd.DataFrame,
                                        time_analysis: pd.DataFrame) -> Dict:
        """
        Génère des visualisations pour l'analyse des décisions
        
        Args:
            decisions_df: DataFrame avec toutes les décisions
            direction_analysis: Analyse par direction
            symbol_analysis: Analyse par symbole
            confidence_analysis: Analyse par niveau de confiance
            time_analysis: Analyse temporelle
            
        Returns:
            Dictionnaire des chemins des visualisations générées
        """
        visualization_paths = {}
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # 1. Performance par direction
        try:
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
            
            # Graphique de win rate
            bar_colors = ['g' if x > 50 else 'r' for x in direction_analysis["win_rate"]]
            ax1.bar(direction_analysis["direction"], direction_analysis["win_rate"], color=bar_colors)
            ax1.set_title("Taux de réussite par direction")
            ax1.set_xlabel("Direction")
            ax1.set_ylabel("Taux de réussite (%)")
            ax1.set_ylim(0, 100)
            
            for i, v in enumerate(direction_analysis["win_rate"]):
                ax1.text(i, v + 1, f"{v:.1f}%", ha='center')
            
            # Graphique de PnL
            bar_colors = ['g' if x > 0 else 'r' for x in direction_analysis["avg_pnl"]]
            ax2.bar(direction_analysis["direction"], direction_analysis["avg_pnl"], color=bar_colors)
            ax2.set_title("PnL moyen par direction")
            ax2.set_xlabel("Direction")
            ax2.set_ylabel("PnL moyen")
            
            for i, v in enumerate(direction_analysis["avg_pnl"]):
                ax2.text(i, v + (0.01 if v >= 0 else -0.01), f"{v:.4f}", ha='center')
            
            plt.tight_layout()
            
            direction_path = os.path.join(self.output_dir, f"direction_performance_{timestamp}.png")
            plt.savefig(direction_path, dpi=300)
            plt.close()
            
            visualization_paths["direction_performance"] = direction_path
        
        except Exception as e:
            logger.error(f"Erreur lors de la génération du graphique de performance par direction: {str(e)}")
        
        # 2. Performance par symbole (top 10)
        try:
            # Trier par nombre de trades et prendre les 10 premiers
            top_symbols = symbol_analysis.sort_values("count", ascending=False).head(10)
            
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
            
            # Graphique de win rate
            bar_colors = ['g' if x > 50 else 'r' for x in top_symbols["win_rate"]]
            ax1.bar(top_symbols["symbol"], top_symbols["win_rate"], color=bar_colors)
            ax1.set_title("Taux de réussite par symbole (Top 10)")
            ax1.set_xlabel("Symbole")
            ax1.set_ylabel("Taux de réussite (%)")
            ax1.set_ylim(0, 100)
            plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45, ha='right')
            
            for i, v in enumerate(top_symbols["win_rate"]):
                ax1.text(i, v + 1, f"{v:.1f}%", ha='center')
            
            # Graphique de PnL total
            bar_colors = ['g' if x > 0 else 'r' for x in top_symbols["total_pnl"]]
            ax2.bar(top_symbols["symbol"], top_symbols["total_pnl"], color=bar_colors)
            ax2.set_title("PnL total par symbole (Top 10)")
            ax2.set_xlabel("Symbole")
            ax2.set_ylabel("PnL total")
            plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45, ha='right')
            
            for i, v in enumerate(top_symbols["total_pnl"]):
                ax2.text(i, v + (0.01 if v >= 0 else -0.01), f"{v:.2f}", ha='center')
            
            plt.tight_layout()
            
            symbol_path = os.path.join(self.output_dir, f"symbol_performance_{timestamp}.png")
            plt.savefig(symbol_path, dpi=300)
            plt.close()
            
            visualization_paths["symbol_performance"] = symbol_path
        
        except Exception as e:
            logger.error(f"Erreur lors de la génération du graphique de performance par symbole: {str(e)}")
        
        # 3. Performance par niveau de confiance
        try:
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
            
            # Graphique de win rate
            ax1.bar(confidence_analysis["confidence_range"], confidence_analysis["win_rate"])
            ax1.set_title("Taux de réussite par niveau de confiance")
            ax1.set_xlabel("Niveau de confiance")
            ax1.set_ylabel("Taux de réussite (%)")
            ax1.set_ylim(0, 100)
            
            for i, v in enumerate(confidence_analysis["win_rate"]):
                ax1.text(i, v + 1, f"{v:.1f}%", ha='center')
            
            # Graphique de count
            ax2.bar(confidence_analysis["confidence_range"], confidence_analysis["count"])
            ax2.set_title("Nombre de décisions par niveau de confiance")
            ax2.set_xlabel("Niveau de confiance")
            ax2.set_ylabel("Nombre de décisions")
            
            for i, v in enumerate(confidence_analysis["count"]):
                ax2.text(i, v + 1, str(v), ha='center')
            
            plt.tight_layout()
            
            confidence_path = os.path.join(self.output_dir, f"confidence_performance_{timestamp}.png")
            plt.savefig(confidence_path, dpi=300)
            plt.close()
            
            visualization_paths["confidence_performance"] = confidence_path
        
        except Exception as e:
            logger.error(f"Erreur lors de la génération du graphique de performance par niveau de confiance: {str(e)}")
        
        # 4. Évolution temporelle
        try:
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), sharex=True)
            
            # Graphique de win rate
            ax1.plot(time_analysis["week_start"], time_analysis["win_rate"], marker='o', linestyle='-')
            ax1.axhline(y=50, color='r', linestyle='--', alpha=0.5)
            ax1.set_title("Évolution du taux de réussite")
            ax1.set_ylabel("Taux de réussite (%)")
            ax1.grid(True, alpha=0.3)
            
            # Formater l'axe des dates
            ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
            ax1.xaxis.set_major_locator(mdates.MonthLocator())
            
            # Graphique de PnL cumulatif
            ax2.plot(time_analysis["week_start"], time_analysis["cumulative_pnl"], marker='o', linestyle='-', color='g')
            ax2.axhline(y=0, color='r', linestyle='--', alpha=0.5)
            ax2.set_title("Évolution du PnL cumulatif")
            ax2.set_xlabel("Semaine")
            ax2.set_ylabel("PnL cumulatif")
            ax2.grid(True, alpha=0.3)
            
            plt.xticks(rotation=45, ha='right')
            plt.tight_layout()
            
            time_path = os.path.join(self.output_dir, f"time_performance_{timestamp}.png")
            plt.savefig(time_path, dpi=300)
            plt.close()
            
            visualization_paths["time_performance"] = time_path
        
        except Exception as e:
            logger.error(f"Erreur lors de la génération du graphique d'évolution temporelle: {str(e)}")
        
        # 5. Distribution des PnL
        try:
            plt.figure(figsize=(10, 6))
            
            sns.histplot(decisions_df["pnl"], bins=30, kde=True)
            plt.axvline(x=0, color='r', linestyle='--')
            plt.title("Distribution des PnL")
            plt.xlabel("PnL")
            plt.ylabel("Fréquence")
            plt.grid(True, alpha=0.3)
            
            pnl_path = os.path.join(self.output_dir, f"pnl_distribution_{timestamp}.png")
            plt.savefig(pnl_path, dpi=300)
            plt.close()
            
            visualization_paths["pnl_distribution"] = pnl_path
        
        except Exception as e:
            logger.error(f"Erreur lors de la génération du graphique de distribution des PnL: {str(e)}")
        
        # 6. Importance des facteurs (poids)
        try:
            # Statistiques des poids
            weight_stats = decisions_df[["strategy_weight", "risk_weight", 
                                       "sentiment_weight", "technical_weight"]].describe()
            
            # Moyennes pour les décisions réussies vs échouées
            successful_weights = decisions_df[decisions_df["success"] == True][
                ["strategy_weight", "risk_weight", "sentiment_weight", "technical_weight"]
            ].mean()
            
            failed_weights = decisions_df[decisions_df["success"] == False][
                ["strategy_weight", "risk_weight", "sentiment_weight", "technical_weight"]
            ].mean()
            
            # Combiner en un DataFrame pour le plotting
            weight_comparison = pd.DataFrame({
                'Tous': weight_stats.loc['mean'],
                'Réussis': successful_weights,
                'Échoués': failed_weights
            }).reset_index()
            weight_comparison.columns = ['Facteur', 'Tous', 'Réussis', 'Échoués']
            
            # Renommer les facteurs pour l'affichage
            factor_names = {
                'strategy_weight': 'Stratégie',
                'risk_weight': 'Risque',
                'sentiment_weight': 'Sentiment',
                'technical_weight': 'Technique'
            }
            weight_comparison['Facteur'] = weight_comparison['Facteur'].map(factor_names)
            
            # Créer le graphique
            plt.figure(figsize=(10, 6))
            
            bar_width = 0.25
            x = np.arange(len(weight_comparison['Facteur']))
            
            plt.bar(x - bar_width, weight_comparison['Tous'], width=bar_width, label='Tous', color='blue', alpha=0.7)
            plt.bar(x, weight_comparison['Réussis'], width=bar_width, label='Réussis', color='green', alpha=0.7)
            plt.bar(x + bar_width, weight_comparison['Échoués'], width=bar_width, label='Échoués', color='red', alpha=0.7)
            
            plt.xlabel('Facteur')
            plt.ylabel('Poids moyen (%)')
            plt.title('Importance des facteurs par résultat de décision')
            plt.xticks(x, weight_comparison['Facteur'])
            plt.legend()
            plt.grid(True, alpha=0.3)
            
            # Convertir les valeurs en pourcentages pour l'affichage
            for i, v in enumerate(weight_comparison['Tous']):
                plt.text(i - bar_width, v + 0.01, f"{v*100:.1f}%", ha='center', fontsize=8)
            
            for i, v in enumerate(weight_comparison['Réussis']):
                plt.text(i, v + 0.01, f"{v*100:.1f}%", ha='center', fontsize=8)
                
            for i, v in enumerate(weight_comparison['Échoués']):
                plt.text(i + bar_width, v + 0.01, f"{v*100:.1f}%", ha='center', fontsize=8)
            
            weights_path = os.path.join(self.output_dir, f"weight_importance_{timestamp}.png")
            plt.savefig(weights_path, dpi=300)
            plt.close()
            
            visualization_paths["weight_importance"] = weights_path
        
        except Exception as e:
            logger.error(f"Erreur lors de la génération du graphique d'importance des facteurs: {str(e)}")
        
        return visualization_paths
    
    def analyze_trades(self, trades: List[Dict]) -> Dict:
        """
        Analyse les trades exécutés
        
        Args:
            trades: Liste des trades
            
        Returns:
            Résultats de l'analyse
        """
        if not trades:
            return {
                "success": False,
                "error": "Aucun trade à analyser"
            }
        
        # Convertir en DataFrame pour faciliter l'analyse
        try:
            trades_df = pd.DataFrame(trades)
            
            # Convertir les timestamps en datetime
            if "open_time" in trades_df.columns:
                trades_df["open_time"] = pd.to_datetime(trades_df["open_time"])
            if "close_time" in trades_df.columns:
                trades_df["close_time"] = pd.to_datetime(trades_df["close_time"])
            
            # Calculer la durée des trades
            if "open_time" in trades_df.columns and "close_time" in trades_df.columns:
                closed_trades = trades_df[~trades_df["close_time"].isna()]
                if not closed_trades.empty:
                    closed_trades["duration"] = (closed_trades["close_time"] - closed_trades["open_time"]).dt.total_seconds() / 3600  # en heures
                    trades_df.loc[closed_trades.index, "duration"] = closed_trades["duration"]
            
            # Analyser les trades par direction
            direction_analysis = None
            if "direction" in trades_df.columns and "pnl" in trades_df.columns:
                direction_analysis = trades_df.groupby("direction").agg(
                    count=("direction", "size"),
                    win_count=("pnl", lambda x: sum(x > 0)),
                    loss_count=("pnl", lambda x: sum(x < 0)),
                    win_rate=("pnl", lambda x: sum(x > 0) / len(x) * 100 if len(x) > 0 else 0),
                    avg_pnl=("pnl", "mean"),
                    total_pnl=("pnl", "sum"),
                    max_pnl=("pnl", "max"),
                    min_pnl=("pnl", "min")
                ).reset_index()
            
            # Analyser les trades par symbole
            symbol_analysis = None
            if "symbol" in trades_df.columns and "pnl" in trades_df.columns:
                symbol_analysis = trades_df.groupby("symbol").agg(
                    count=("symbol", "size"),
                    win_count=("pnl", lambda x: sum(x > 0)),
                    loss_count=("pnl", lambda x: sum(x < 0)),
                    win_rate=("pnl", lambda x: sum(x > 0) / len(x) * 100 if len(x) > 0 else 0),
                    avg_pnl=("pnl", "mean"),
                    total_pnl=("pnl", "sum"),
                    avg_duration=("duration", "mean")
                ).reset_index()
            
            # Analyser les trades par jour de la semaine
            weekday_analysis = None
            if "open_time" in trades_df.columns and "pnl" in trades_df.columns:
                trades_df["weekday"] = trades_df["open_time"].dt.day_name()
                weekday_analysis = trades_df.groupby("weekday").agg(
                    count=("weekday", "size"),
                    win_rate=("pnl", lambda x: sum(x > 0) / len(x) * 100 if len(x) > 0 else 0),
                    avg_pnl=("pnl", "mean"),
                    total_pnl=("pnl", "sum")
                ).reset_index()
            
            # Analyser les trades par heure
            hourly_analysis = None
            if "open_time" in trades_df.columns and "pnl" in trades_df.columns:
                trades_df["hour"] = trades_df["open_time"].dt.hour
                hourly_analysis = trades_df.groupby("hour").agg(
                    count=("hour", "size"),
                    win_rate=("pnl", lambda x: sum(x > 0) / len(x) * 100 if len(x) > 0 else 0),
                    avg_pnl=("pnl", "mean"),
                    total_pnl=("pnl", "sum")
                ).reset_index()
            
            # Analyser l'évolution des performances dans le temps
            time_analysis = None
            if "open_time" in trades_df.columns and "pnl" in trades_df.columns:
                trades_df["date"] = trades_df["open_time"].dt.date
                daily_pnl = trades_df.groupby("date")["pnl"].sum().reset_index()
                daily_pnl["cumulative_pnl"] = daily_pnl["pnl"].cumsum()
                
                time_analysis = {
                    "dates": daily_pnl["date"].astype(str).tolist(),
                    "daily_pnl": daily_pnl["pnl"].tolist(),
                    "cumulative_pnl": daily_pnl["cumulative_pnl"].tolist()
                }
            
            # Calculer les métriques de performance
            metrics = {
                "total_trades": len(trades_df),
                "winning_trades": sum(trades_df["pnl"] > 0) if "pnl" in trades_df.columns else None,
                "losing_trades": sum(trades_df["pnl"] < 0) if "pnl" in trades_df.columns else None,
                "win_rate": sum(trades_df["pnl"] > 0) / len(trades_df) * 100 if "pnl" in trades_df.columns else None,
                "avg_pnl": trades_df["pnl"].mean() if "pnl" in trades_df.columns else None,
                "total_pnl": trades_df["pnl"].sum() if "pnl" in trades_df.columns else None,
                "max_win": trades_df["pnl"].max() if "pnl" in trades_df.columns else None,
                "max_loss": trades_df["pnl"].min() if "pnl" in trades_df.columns else None,
                "avg_duration": trades_df["duration"].mean() if "duration" in trades_df.columns else None
            }
            
            # Calculer des ratios de performance supplémentaires
            if "pnl" in trades_df.columns:
                winning_trades = trades_df[trades_df["pnl"] > 0]
                losing_trades = trades_df[trades_df["pnl"] < 0]
                
                # Profit factor (somme des gains / somme des pertes en valeur absolue)
                if not losing_trades.empty and losing_trades["pnl"].sum() != 0:
                    metrics["profit_factor"] = winning_trades["pnl"].sum() / abs(losing_trades["pnl"].sum())
                else:
                    metrics["profit_factor"] = float("inf") if not winning_trades.empty else 0
                
                # Ratio gain/perte moyen
                if not losing_trades.empty and losing_trades["pnl"].mean() != 0:
                    metrics["win_loss_ratio"] = winning_trades["pnl"].mean() / abs(losing_trades["pnl"].mean()) if not winning_trades.empty else 0
                else:
                    metrics["win_loss_ratio"] = float("inf") if not winning_trades.empty else 0
                
                # Expectancy (espérance mathématique par trade)
                if len(trades_df) > 0:
                    win_rate = metrics["winning_trades"] / len(trades_df)
                    lose_rate = metrics["losing_trades"] / len(trades_df)
                    avg_win = winning_trades["pnl"].mean() if not winning_trades.empty else 0
                    avg_loss = losing_trades["pnl"].mean() if not losing_trades.empty else 0
                    
                    metrics["expectancy"] = (win_rate * avg_win) + (lose_rate * avg_loss)
                
                # Ratio de Sharpe simplifié (supposant un taux sans risque de 0)
                if len(trades_df) > 10 and trades_df["pnl"].std() > 0:
                    metrics["sharpe_ratio"] = (trades_df["pnl"].mean() / trades_df["pnl"].std()) * np.sqrt(252)  # Annualisé
            
            # Générer des visualisations
            visualization_paths = {}
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            # 1. Courbe de PnL cumulatif
            if time_analysis:
                try:
                    plt.figure(figsize=(12, 6))
                    plt.plot(pd.to_datetime(daily_pnl["date"]), daily_pnl["cumulative_pnl"], marker='o')
                    plt.axhline(y=0, color='r', linestyle='--')
                    plt.title("PnL cumulatif")
                    plt.xlabel("Date")
                    plt.ylabel("PnL")
                    plt.grid(True, alpha=0.3)
                    plt.xticks(rotation=45)
                    
                    pnl_path = os.path.join(self.output_dir, f"cumulative_pnl_{timestamp}.png")
                    plt.savefig(pnl_path, dpi=300)
                    plt.close()
                    
                    visualization_paths["cumulative_pnl"] = pnl_path
                except Exception as e:
                    logger.error(f"Erreur lors de la génération du graphique de PnL cumulatif: {str(e)}")
            
            # 2. Distribution des PnL par trade
            if "pnl" in trades_df.columns:
                try:
                    plt.figure(figsize=(10, 6))
                    sns.histplot(trades_df["pnl"], bins=30, kde=True)
                    plt.axvline(x=0, color='r', linestyle='--')
                    plt.title("Distribution des PnL par trade")
                    plt.xlabel("PnL")
                    plt.ylabel("Fréquence")
                    
                    pnl_dist_path = os.path.join(self.output_dir, f"pnl_distribution_trades_{timestamp}.png")
                    plt.savefig(pnl_dist_path, dpi=300)
                    plt.close()
                    
                    visualization_paths["pnl_distribution"] = pnl_dist_path
                except Exception as e:
                    logger.error(f"Erreur lors de la génération du graphique de distribution des PnL: {str(e)}")
            
            # 3. Performance par jour de la semaine
            if weekday_analysis is not None:
                try:
                    # Réorganiser les jours dans l'ordre
                    day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
                    weekday_analysis['weekday'] = pd.Categorical(weekday_analysis['weekday'], categories=day_order, ordered=True)
                    weekday_analysis = weekday_analysis.sort_values('weekday')
                    
                    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
                    
                    # Win rate par jour
                    ax1.bar(weekday_analysis["weekday"], weekday_analysis["win_rate"])
                    ax1.set_title("Taux de réussite par jour de la semaine")
                    ax1.set_xlabel("Jour")
                    ax1.set_ylabel("Taux de réussite (%)")
                    ax1.set_ylim(0, 100)
                    plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45, ha='right')
                    
                    # Nombre de trades par jour
                    ax2.bar(weekday_analysis["weekday"], weekday_analysis["count"])
                    ax2.set_title("Nombre de trades par jour")
                    ax2.set_xlabel("Jour")
                    ax2.set_ylabel("Nombre de trades")
                    plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45, ha='right')
                    
                    plt.tight_layout()
                    
                    weekday_path = os.path.join(self.output_dir, f"weekday_performance_{timestamp}.png")
                    plt.savefig(weekday_path, dpi=300)
                    plt.close()
                    
                    visualization_paths["weekday_performance"] = weekday_path
                except Exception as e:
                    logger.error(f"Erreur lors de la génération du graphique par jour de semaine: {str(e)}")
            
            # 4. Performance par heure
            if hourly_analysis is not None:
                try:
                    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
                    
                    # Win rate par heure
                    ax1.bar(hourly_analysis["hour"], hourly_analysis["win_rate"])
                    ax1.set_title("Taux de réussite par heure")
                    ax1.set_xlabel("Heure")
                    ax1.set_ylabel("Taux de réussite (%)")
                    ax1.set_ylim(0, 100)
                    ax1.set_xticks(range(0, 24, 2))
                    
                    # PnL moyen par heure
                    ax2.bar(hourly_analysis["hour"], hourly_analysis["avg_pnl"])
                    ax2.set_title("PnL moyen par heure")
                    ax2.set_xlabel("Heure")
                    ax2.set_ylabel("PnL moyen")
                    ax2.set_xticks(range(0, 24, 2))
                    
                    plt.tight_layout()
                    
                    hourly_path = os.path.join(self.output_dir, f"hourly_performance_{timestamp}.png")
                    plt.savefig(hourly_path, dpi=300)
                    plt.close()
                    
                    visualization_paths["hourly_performance"] = hourly_path
                except Exception as e:
                    logger.error(f"Erreur lors de la génération du graphique par heure: {str(e)}")
            
            # 5. Performance par symbole (Top 10)
            if symbol_analysis is not None and len(symbol_analysis) > 0:
                try:
                    # Trier par nombre de trades et prendre les 10 premiers
                    top_symbols = symbol_analysis.sort_values("count", ascending=False).head(10)
                    
                    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
                    
                    # Win rate par symbole
                    bar_colors = ['g' if x > 50 else 'r' for x in top_symbols["win_rate"]]
                    ax1.bar(top_symbols["symbol"], top_symbols["win_rate"], color=bar_colors)
                    ax1.set_title("Taux de réussite par symbole (Top 10)")
                    ax1.set_xlabel("Symbole")
                    ax1.set_ylabel("Taux de réussite (%)")
                    ax1.set_ylim(0, 100)
                    plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45, ha='right')
                    
                    # PnL total par symbole
                    bar_colors = ['g' if x > 0 else 'r' for x in top_symbols["total_pnl"]]
                    ax2.bar(top_symbols["symbol"], top_symbols["total_pnl"], color=bar_colors)
                    ax2.set_title("PnL total par symbole (Top 10)")
                    ax2.set_xlabel("Symbole")
                    ax2.set_ylabel("PnL total")
                    plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45, ha='right')
                    
                    plt.tight_layout()
                    
                    symbol_path = os.path.join(self.output_dir, f"symbol_trade_performance_{timestamp}.png")
                    plt.savefig(symbol_path, dpi=300)
                    plt.close()
                    
                    visualization_paths["symbol_performance"] = symbol_path
                except Exception as e:
                    logger.error(f"Erreur lors de la génération du graphique par symbole: {str(e)}")
            
            # Créer et retourner le rapport d'analyse
            report = {
                "success": True,
                "analysis_timestamp": datetime.now().isoformat(),
                "total_trades_analyzed": len(trades_df),
                "metrics": metrics,
                "direction_analysis": direction_analysis.to_dict(orient="records") if direction_analysis is not None else None,
                "symbol_analysis": symbol_analysis.to_dict(orient="records") if symbol_analysis is not None else None,
                "weekday_analysis": weekday_analysis.to_dict(orient="records") if weekday_analysis is not None else None,
                "hourly_analysis": hourly_analysis.to_dict(orient="records") if hourly_analysis is not None else None,
                "time_analysis": time_analysis,
                "visualization_paths": visualization_paths
            }
            
            # Sauvegarder le rapport
            report_path = os.path.join(self.output_dir, f"trades_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
            try:
                with open(report_path, 'w') as f:
                    json.dump(report, f, indent=2, default=str)
                report["report_path"] = report_path
            except Exception as e:
                logger.error(f"Erreur lors de la sauvegarde du rapport de trades: {str(e)}")
            
            return report
            
        except Exception as e:
            logger.error(f"Erreur lors de l'analyse des trades: {str(e)}")
            return {
                "success": False,
                "error": str(e)
            }
    
    def generate_performance_dashboard(self, decision_history: Dict, trades: List[Dict], 
                                   output_file: str = None, include_plotly: bool = True) -> str:
        """
        Génère un tableau de bord de performance interactif au format HTML
        
        Args:
            decision_history: Historique des décisions
            trades: Liste des trades exécutés
            output_file: Chemin du fichier de sortie (généré automatiquement si None)
            include_plotly: Inclure des graphiques interactifs Plotly
            
        Returns:
            Chemin du tableau de bord généré
        """
        if not output_file:
            output_file = os.path.join(self.output_dir, f"performance_dashboard_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html")
        
        # Analyser les décisions et les trades
        decision_analysis = self.analyze_decision_engine(decision_history, {})
        trades_analysis = self.analyze_trades(trades)
        
        # Vérifier si les analyses ont réussi
        if not decision_analysis.get("success", False) or not trades_analysis.get("success", False):
            error_msg = "Erreur lors de l'analyse des données pour le tableau de bord"
            logger.error(error_msg)
            return None
        
        # Préparer le contenu HTML
        html_content = [
            "<!DOCTYPE html>",
            "<html lang='fr'>",
            "<head>",
            "    <meta charset='UTF-8'>",
            "    <meta name='viewport' content='width=device-width, initial-scale=1.0'>",
            "    <title>Tableau de Bord de Performance - Bot de Trading</title>",
            "    <style>",
            "        body { font-family: Arial, sans-serif; margin: 0; padding: 20px; background-color: #f5f5f5; }",
            "        .container { max-width: 1200px; margin: 0 auto; }",
            "        .dashboard-header { text-align: center; margin-bottom: 30px; }",
            "        .dashboard-section { background-color: white; padding: 20px; margin-bottom: 20px; border-radius: 5px; box-shadow: 0 2px 5px rgba(0,0,0,0.1); }",
            "        .metrics-grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(200px, 1fr)); gap: 20px; }",
            "        .metric-card { background-color: #f9f9f9; padding: 15px; border-radius: 5px; text-align: center; }",
            "        .metric-value { font-size: 24px; font-weight: bold; margin: 10px 0; }",
            "        .metric-label { font-size: 14px; color: #666; }",
            "        .positive { color: green; }",
            "        .negative { color: red; }",
            "        .neutral { color: blue; }",
            "        .chart-container { margin: 20px 0; }",
            "        table { width: 100%; border-collapse: collapse; margin: 20px 0; }",
            "        th, td { padding: 12px 15px; text-align: left; border-bottom: 1px solid #ddd; }",
            "        th { background-color: #f2f2f2; }",
            "        tr:hover { background-color: #f5f5f5; }",
            "        .tabs { display: flex; margin-bottom: -1px; }",
            "        .tab { padding: 10px 20px; cursor: pointer; border: 1px solid #ddd; background: #f1f1f1; }",
            "        .tab.active { background: white; border-bottom: 1px solid white; }",
            "        .tab-content { display: none; padding: 20px; border: 1px solid #ddd; }",
            "        .tab-content.active { display: block; }",
            "    </style>"
        ]
        
        # Inclure Plotly.js si demandé
        if include_plotly:
            html_content.append("    <script src='https://cdn.plot.ly/plotly-latest.min.js'></script>")
        
        # Fermer le head et commencer le body
        html_content.extend([
            "</head>",
            "<body>",
            "    <div class='container'>",
            "        <div class='dashboard-header'>",
            f"            <h1>Tableau de Bord de Performance</h1>",
            f"            <p>Généré le {datetime.now().strftime('%d-%m-%Y à %H:%M:%S')}</p>",
            "        </div>"
        ])
        
        # Section des métriques générales
        html_content.extend([
            "        <div class='dashboard-section'>",
            "            <h2>Métriques Générales</h2>",
            "            <div class='metrics-grid'>"
        ])
        
        # Métriques des trades
        trade_metrics = trades_analysis.get("metrics", {})
        if trade_metrics:
            win_rate = trade_metrics.get("win_rate", 0)
            html_content.extend([
                "                <div class='metric-card'>",
                "                    <div class='metric-label'>Trades Totaux</div>",
                f"                    <div class='metric-value'>{trade_metrics.get('total_trades', 0)}</div>",
                "                </div>",
                "                <div class='metric-card'>",
                "                    <div class='metric-label'>Taux de Réussite</div>",
                f"                    <div class='metric-value {self._get_color_class(win_rate, 50)}'>{win_rate:.2f}%</div>",
                "                </div>",
                "                <div class='metric-card'>",
                "                    <div class='metric-label'>PnL Total</div>",
                f"                    <div class='metric-value {self._get_color_class(trade_metrics.get('total_pnl', 0), 0)}'>{trade_metrics.get('total_pnl', 0):.4f}</div>",
                "                </div>",
                "                <div class='metric-card'>",
                "                    <div class='metric-label'>PnL Moyen</div>",
                f"                    <div class='metric-value {self._get_color_class(trade_metrics.get('avg_pnl', 0), 0)}'>{trade_metrics.get('avg_pnl', 0):.4f}</div>",
                "                </div>",
                "                <div class='metric-card'>",
                "                    <div class='metric-label'>Profit Factor</div>",
                f"                    <div class='metric-value {self._get_color_class(trade_metrics.get('profit_factor', 0), 1)}'>{trade_metrics.get('profit_factor', 0):.2f}</div>",
                "                </div>",
                "                <div class='metric-card'>",
                "                    <div class='metric-label'>Ratio Sharpe</div>",
                f"                    <div class='metric-value {self._get_color_class(trade_metrics.get('sharpe_ratio', 0), 1)}'>{trade_metrics.get('sharpe_ratio', 0):.2f}</div>",
                "                </div>"
            ])
        
        # Métriques des décisions
        decision_metrics = decision_analysis.get("metrics", {})
        if decision_metrics:
            html_content.extend([
                "                <div class='metric-card'>",
                "                    <div class='metric-label'>Décisions Totales</div>",
                f"                    <div class='metric-value'>{decision_metrics.get('total_decisions', 0)}</div>",
                "                </div>",
                "                <div class='metric-card'>",
                "                    <div class='metric-label'>Précision des Décisions</div>",
                f"                    <div class='metric-value {self._get_color_class(decision_metrics.get('win_rate', 0), 50)}'>{decision_metrics.get('win_rate', 0):.2f}%</div>",
                "                </div>"
            ])
        
        html_content.append("            </div>")  # Fermer metrics-grid
        
        # Ajouter des graphiques interactifs avec Plotly
        if include_plotly:
            html_content.append("            <div class='chart-container'>")
            
            # Graphique de PnL cumulatif
            time_analysis = trades_analysis.get("time_analysis", {})
            if time_analysis and "dates" in time_analysis and "cumulative_pnl" in time_analysis:
                html_content.extend([
                    "                <div id='cumulative_pnl_chart' style='width:100%; height:400px;'></div>",
                    "                <script>",
                    "                    var trace = {",
                    f"                        x: {json.dumps(time_analysis['dates'])},",
                    f"                        y: {json.dumps(time_analysis['cumulative_pnl'])},",
                    "                        type: 'scatter',",
                    "                        mode: 'lines+markers',",
                    "                        line: {color: '#2CA02C'},",
                    "                        name: 'PnL Cumulatif'",
                    "                    };",
                    "                    var layout = {",
                    "                        title: 'Évolution du PnL Cumulatif',",
                    "                        xaxis: {title: 'Date'},",
                    "                        yaxis: {title: 'PnL'}",
                    "                    };",
                    "                    Plotly.newPlot('cumulative_pnl_chart', [trace], layout);",
                    "                </script>"
                ])
            
            # Graphique de performance par jour
            weekday_analysis = trades_analysis.get("weekday_analysis", [])
            if weekday_analysis:
                # Convertir en listes pour Plotly
                days = []
                win_rates = []
                counts = []
                
                # Ordre des jours
                day_order = {'Monday': 0, 'Tuesday': 1, 'Wednesday': 2, 'Thursday': 3, 
                        'Friday': 4, 'Saturday': 5, 'Sunday': 6}
                
                # Trier par jour
                sorted_analysis = sorted(weekday_analysis, key=lambda x: day_order.get(x['weekday'], 7))
                
                for day in sorted_analysis:
                    days.append(day['weekday'])
                    win_rates.append(day['win_rate'])
                    counts.append(day['count'])
                
                html_content.extend([
                    "                <div id='weekday_chart' style='width:100%; height:400px;'></div>",
                    "                <script>",
                    "                    var trace1 = {",
                    f"                        x: {json.dumps(days)},",
                    f"                        y: {json.dumps(win_rates)},",
                    "                        type: 'bar',",
                    "                        name: 'Taux de Réussite (%)',",
                    "                        yaxis: 'y',",
                    "                        marker: {color: '#1F77B4'}",
                    "                    };",
                    "                    var trace2 = {",
                    f"                        x: {json.dumps(days)},",
                    f"                        y: {json.dumps(counts)},",
                    "                        type: 'bar',",
                    "                        name: 'Nombre de Trades',",
                    "                        yaxis: 'y2',",
                    "                        marker: {color: '#FF7F0E'}",
                    "                    };",
                    "                    var layout = {",
                    "                        title: 'Performance par Jour de la Semaine',",
                    "                        yaxis: {title: 'Taux de Réussite (%)', side: 'left'},",
                    "                        yaxis2: {title: 'Nombre de Trades', side: 'right', overlaying: 'y'},",
                    "                        barmode: 'group'",
                    "                    };",
                    "                    Plotly.newPlot('weekday_chart', [trace1, trace2], layout);",
                    "                </script>"
                ])
            
            html_content.append("            </div>")  # Fermer chart-container
        
        html_content.append("        </div>")  # Fermer dashboard-section
        
        # Ajouter des onglets pour différentes analyses
        html_content.extend([
            "        <div class='dashboard-section'>",
            "            <div class='tabs'>",
            "                <div class='tab active' onclick=\"showTab('tab-trades')\">Trades</div>",
            "                <div class='tab' onclick=\"showTab('tab-symbols')\">Symboles</div>",
            "                <div class='tab' onclick=\"showTab('tab-time')\">Analyse Temporelle</div>",
            "                <div class='tab' onclick=\"showTab('tab-decisions')\">Décisions</div>",
            "            </div>"
        ])
        
        # Contenu de l'onglet Trades
        html_content.extend([
            "            <div id='tab-trades' class='tab-content active'>",
            "                <h3>Analyse des Trades par Direction</h3>"
        ])
        
        # Tableau des trades par direction
        direction_analysis = trades_analysis.get("direction_analysis", [])
        if direction_analysis:
            html_content.extend([
                "                <table>",
                "                    <thead>",
                "                        <tr>",
                "                            <th>Direction</th>",
                "                            <th>Nombre</th>",
                "                            <th>Gagnants</th>",
                "                            <th>Perdants</th>",
                "                            <th>Taux de Réussite</th>",
                "                            <th>PnL Moyen</th>",
                "                            <th>PnL Total</th>",
                "                        </tr>",
                "                    </thead>",
                "                    <tbody>"
            ])
            
            for direction in direction_analysis:
                html_content.append(f"                        <tr>")
                html_content.append(f"                            <td>{direction['direction']}</td>")
                html_content.append(f"                            <td>{direction['count']}</td>")
                html_content.append(f"                            <td>{direction['win_count']}</td>")
                html_content.append(f"                            <td>{direction['loss_count']}</td>")
                win_rate = direction['win_rate']
                win_rate_class = 'positive' if win_rate >= 50 else 'negative'
                html_content.append(f"                            <td class='{win_rate_class}'>{win_rate:.2f}%</td>")
                
                avg_pnl = direction['avg_pnl']
                avg_pnl_class = 'positive' if avg_pnl > 0 else 'negative' if avg_pnl < 0 else ''
                html_content.append(f"                            <td class='{avg_pnl_class}'>{avg_pnl:.4f}</td>")
                
                total_pnl = direction['total_pnl']
                total_pnl_class = 'positive' if total_pnl > 0 else 'negative' if total_pnl < 0 else ''
                html_content.append(f"                            <td class='{total_pnl_class}'>{total_pnl:.4f}</td>")
                html_content.append(f"                        </tr>")
            
            html_content.append("                    </tbody>")
            html_content.append("                </table>")
        else:
            html_content.append("                <p>Aucune donnée d'analyse de direction disponible.</p>")
        
        html_content.append("            </div>")  # Fermer tab-trades
        
        # Contenu de l'onglet Symboles
        html_content.extend([
            "            <div id='tab-symbols' class='tab-content'>",
            "                <h3>Analyse des Trades par Symbole</h3>"
        ])
        
        # Tableau des trades par symbole
        symbol_analysis = trades_analysis.get("symbol_analysis", [])
        if symbol_analysis:
            # Trier par nombre de trades
            symbol_analysis = sorted(symbol_analysis, key=lambda x: x['count'], reverse=True)
            
            html_content.extend([
                "                <table>",
                "                    <thead>",
                "                        <tr>",
                "                            <th>Symbole</th>",
                "                            <th>Nombre</th>",
                "                            <th>Gagnants</th>",
                "                            <th>Perdants</th>",
                "                            <th>Taux de Réussite</th>",
                "                            <th>PnL Moyen</th>",
                "                            <th>PnL Total</th>",
                "                            <th>Durée Moyenne (heures)</th>",
                "                        </tr>",
                "                    </thead>",
                "                    <tbody>"
            ])
            
            for symbol in symbol_analysis:
                html_content.append(f"                        <tr>")
                html_content.append(f"                            <td>{symbol['symbol']}</td>")
                html_content.append(f"                            <td>{symbol['count']}</td>")
                html_content.append(f"                            <td>{symbol['win_count']}</td>")
                html_content.append(f"                            <td>{symbol['loss_count']}</td>")
                win_rate = symbol['win_rate']
                win_rate_class = 'positive' if win_rate >= 50 else 'negative'
                html_content.append(f"                            <td class='{win_rate_class}'>{win_rate:.2f}%</td>")
                
                avg_pnl = symbol['avg_pnl']
                avg_pnl_class = 'positive' if avg_pnl > 0 else 'negative' if avg_pnl < 0 else ''
                html_content.append(f"                            <td class='{avg_pnl_class}'>{avg_pnl:.4f}</td>")
                
                total_pnl = symbol['total_pnl']
                total_pnl_class = 'positive' if total_pnl > 0 else 'negative' if total_pnl < 0 else ''
                html_content.append(f"                            <td class='{total_pnl_class}'>{total_pnl:.4f}</td>")
                
                avg_duration = symbol['avg_duration']
                avg_duration_class = 'positive' if avg_duration > 0 else 'negative' if avg_duration < 0 else ''
                html_content.append(f"                            <td class='{avg_duration_class}'>{avg_duration:.2f}</td>")
                html_content.append(f"                        </tr>")
            
            html_content.append("                    </tbody>")
            html_content.append("                </table>")
        else:
            html_content.append("                <p>Aucune donnée d'analyse de symbole disponible.</p>")
        
        html_content.append("            </div>")  # Fermer tab-symbols
        
        # Contenu de l'onglet Analyse Temporelle
        html_content.extend([
            "            <div id='tab-time' class='tab-content'>",
            "                <h3>Analyse Temporelle des Trades</h3>"
        ])
        
        # Graphique de PnL cumulatif
        if time_analysis:
            html_content.extend([
                "                <div class='chart-container'>",
                "                    <div id='cumulative_pnl_chart_time' style='width:100%; height:400px;'></div>",
                "                    <script>",
                "                        var trace = {",
                f"                            x: {json.dumps(time_analysis['dates'])},",
                f"                            y: {json.dumps(time_analysis['cumulative_pnl'])},",
                "                            type: 'scatter',",
                "                            mode: 'lines+markers',",
                "                            line: {color: '#2CA02C'},",
                "                            name: 'PnL Cumulatif'",
                "                        };",
                "                        var layout = {",
                "                            title: 'Évolution du PnL Cumulatif',",
                "                            xaxis: {title: 'Date'},",
                "                            yaxis: {title: 'PnL'}",
                "                        };",
                "                        Plotly.newPlot('cumulative_pnl_chart_time', [trace], layout);",
                "                    </script>",
                "                </div>"
            ])
        
        html_content.append("            </div>")  # Fermer tab-time
        
        # Contenu de l'onglet Décisions
        html_content.extend([
            "            <div id='tab-decisions' class='tab-content'>",
            "                <h3>Analyse des Décisions</h3>"
        ])
        
        # Tableau des décisions par direction
        decision_direction_analysis = decision_analysis.get("direction_performance", [])
        if decision_direction_analysis:
            html_content.extend([
                "                <h4>Performance par Direction</h4>",
                "                <table>",
                "                    <thead>",
                "                        <tr>",
                "                            <th>Direction</th>",
                "                            <th>Nombre</th>",
                "                            <th>Taux de Réussite</th>",
                "                            <th>PnL Moyen</th>",
                "                            <th>PnL Total</th>",
                "                        </tr>",
                "                    </thead>",
                "                    <tbody>"
            ])
            
            for direction in decision_direction_analysis:
                html_content.append(f"                        <tr>")
                html_content.append(f"                            <td>{direction['direction']}</td>")
                html_content.append(f"                            <td>{direction['count']}</td>")
                win_rate = direction['win_rate']
                win_rate_class = 'positive' if win_rate >= 50 else 'negative'
                html_content.append(f"                            <td class='{win_rate_class}'>{win_rate:.2f}%</td>")
                
                avg_pnl = direction['avg_pnl']
                avg_pnl_class = 'positive' if avg_pnl > 0 else 'negative' if avg_pnl < 0 else ''
                html_content.append(f"                            <td class='{avg_pnl_class}'>{avg_pnl:.4f}</td>")
                
                total_pnl = direction['total_pnl']
                total_pnl_class = 'positive' if total_pnl > 0 else 'negative' if total_pnl < 0 else ''
                html_content.append(f"                            <td class='{total_pnl_class}'>{total_pnl:.4f}</td>")
                html_content.append(f"                        </tr>")
            
            html_content.append("                    </tbody>")
            html_content.append("                </table>")
        else:
            html_content.append("                <p>Aucune donnée d'analyse de direction disponible.</p>")
        
        html_content.append("            </div>")  # Fermer tab-decisions
        
        html_content.append("        </div>")  # Fermer dashboard-section
        
        # Ajouter le script pour les onglets
        html_content.extend([
            "        <script>",
            "            function showTab(tabId) {",
            "                var tabs = document.getElementsByClassName('tab');",
            "                for (var i = 0; i < tabs.length; i++) {",
            "                    tabs[i].classList.remove('active');",
            "                }",
            "                var tabContents = document.getElementsByClassName('tab-content');",
            "                for (var i = 0; i < tabContents.length; i++) {",
            "                    tabContents[i].classList.remove('active');",
            "                }",
            "                document.getElementById(tabId).classList.add('active');",
            "                event.currentTarget.classList.add('active');",
            "            }",
            "        </script>",
            "    </div>",
            "</body>",
            "</html>"
        ])
        
        # Sauvegarder le contenu HTML dans un fichier
        try:
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write('\n'.join(html_content))
            return output_file
        except Exception as e:
            logger.error(f"Erreur lors de la sauvegarde du tableau de bord: {str(e)}")
            return None
    
    def _get_color_class(self, value: float, threshold: float) -> str:
        """
        Retourne la classe de couleur CSS en fonction de la valeur et du seuil
        
        Args:
            value: Valeur à évaluer
            threshold: Seuil pour déterminer la classe de couleur
            
        Returns:
            Classe de couleur CSS
        """
        if value > threshold:
            return 'positive'
        elif value < threshold:
            return 'negative'
        else:
            return 'neutral'

def calculate_sharpe_ratio(equity_curve, risk_free_rate=0.02):
    """
    Calcule le Sharpe ratio à partir d'une courbe d'équité.
    
    Args:
        equity_curve: Liste ou array de valeurs d'équité sur le temps.
        risk_free_rate: Taux sans risque annuel (par défaut 2%).
        
    Returns:
        Sharpe ratio (float).
    """
    # Calculer les rendements journaliers
    equity = np.array(equity_curve)
    daily_returns = np.diff(equity) / equity[:-1]
    avg_return = np.mean(daily_returns)
    std_return = np.std(daily_returns)
    if std_return == 0:
        return 0.0
    # Annualiser le ratio (252 jours de trading)
    sharpe_ratio = (avg_return - risk_free_rate / 252) / std_return * np.sqrt(252)
    return sharpe_ratio

def calculate_max_drawdown(equity_curve):
    """
    Calcule le drawdown maximal à partir de la courbe d'équité.
    
    Args:
        equity_curve: Liste ou array de valeurs d'équité sur le temps.
        
    Returns:
        Drawdown maximal (float, en décimal).
    """
    equity = np.array(equity_curve)
    peak = np.maximum.accumulate(equity)
    drawdown = (peak - equity) / peak
    return np.max(drawdown)

def analyze_performance(equity_curve, trades, initial_capital):
    """
    Analyse la performance d'un backtest en calculant diverses métriques.
    
    Args:
        equity_curve: Liste de valeurs d'équité.
        trades: Liste de dictionnaires de trades (doit contenir une clé 'pnl').
        initial_capital: Capital initial utilisé au début du backtest.
        
    Returns:
        Dictionnaire contenant les métriques de performance.
    """
    final_equity = equity_curve[-1]
    total_return = final_equity - initial_capital
    total_return_pct = (total_return / initial_capital) * 100
    
    sharpe_ratio = calculate_sharpe_ratio(equity_curve)
    max_dd = calculate_max_drawdown(equity_curve)
    
    total_trades = len(trades)
    winning_trades = sum(1 for t in trades if t.get('pnl', 0) > 0)
    losing_trades = total_trades - winning_trades
    win_rate = (winning_trades / total_trades * 100) if total_trades > 0 else 0
    total_pnl = sum(t.get('pnl', 0) for t in trades)
    avg_pnl = (total_pnl / total_trades) if total_trades > 0 else 0
    
    metrics = {
        "final_equity": final_equity,
        "total_return": total_return,
        "total_return_pct": total_return_pct,
        "sharpe_ratio": sharpe_ratio,
        "max_drawdown": max_dd,
        "total_trades": total_trades,
        "winning_trades": winning_trades,
        "losing_trades": losing_trades,
        "win_rate": win_rate,
        "avg_pnl": avg_pnl,
        "analysis_timestamp": datetime.now().isoformat()
    }
    return metrics

if __name__ == "__main__":
    # Exemple d'utilisation avec des données fictives
    equity_curve_example = [10000, 10200, 10150, 10300, 10500, 10400]
    trades_example = [
        {"pnl": 200},
        {"pnl": -50},
        {"pnl": 150},
        {"pnl": -100},
    ]
    metrics = analyze_performance(equity_curve_example, trades_example, 10000)
    print(metrics)