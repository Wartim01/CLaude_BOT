# ai/trade_analyzer.py
"""
Analyseur post-trade pour l'amélioration continue
"""
import os
import json
import logging
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Union
from datetime import datetime, timedelta

from config.config import DATA_DIR
from utils.logger import setup_logger

logger = setup_logger("trade_analyzer")

class TradeAnalyzer:
    """
    Analyse les trades passés pour identifier les patterns de succès et d'échec
    """
    def __init__(self, scoring_engine, position_tracker):
        self.scoring_engine = scoring_engine
        self.position_tracker = position_tracker
        self.analysis_dir = os.path.join(DATA_DIR, "analysis")
        
        # Créer le répertoire si nécessaire
        if not os.path.exists(self.analysis_dir):
            os.makedirs(self.analysis_dir)
    
    def analyze_recent_trades(self, days: int = 7) -> Dict:
        """
        Analyse les trades récents pour identifier les facteurs de succès et d'échec
        
        Args:
            days: Nombre de jours à analyser
            
        Returns:
            Rapport d'analyse
        """
        # Récupérer les trades fermés
        closed_positions = self.position_tracker.get_closed_positions(limit=1000)
        
        # Filtrer sur la période demandée
        start_date = datetime.now() - timedelta(days=days)
        
        filtered_positions = []
        for position in closed_positions:
            # Convertir la date de fermeture si elle est sous forme de chaîne
            close_time = position.get("close_time")
            if isinstance(close_time, str):
                try:
                    close_time = datetime.fromisoformat(close_time)
                except:
                    continue
            
            if close_time and close_time > start_date:
                filtered_positions.append(position)
        
        if not filtered_positions:
            logger.warning(f"Aucun trade fermé dans les {days} derniers jours")
            return {
                "success": False,
                "message": f"Aucun trade fermé dans les {days} derniers jours"
            }
        
        # Analyser les trades
        successful_trades = [p for p in filtered_positions if p.get("pnl_percent", 0) > 0]
        failed_trades = [p for p in filtered_positions if p.get("pnl_percent", 0) <= 0]
        
        # Calculer les statistiques générales
        total_trades = len(filtered_positions)
        success_rate = len(successful_trades) / total_trades * 100 if total_trades > 0 else 0
        
        avg_pnl = sum(p.get("pnl_percent", 0) for p in filtered_positions) / total_trades if total_trades > 0 else 0
        avg_win = sum(p.get("pnl_percent", 0) for p in successful_trades) / len(successful_trades) if successful_trades else 0
        avg_loss = sum(p.get("pnl_percent", 0) for p in failed_trades) / len(failed_trades) if failed_trades else 0
        
        # Calculer la durée moyenne des trades
        durations = []
        for position in filtered_positions:
            entry_time = position.get("entry_time")
            close_time = position.get("close_time")
            
            if entry_time and close_time:
                # Convertir en datetime si nécessaire
                if isinstance(entry_time, str):
                    try:
                        entry_time = datetime.fromisoformat(entry_time)
                    except:
                        continue
                
                if isinstance(close_time, str):
                    try:
                        close_time = datetime.fromisoformat(close_time)
                    except:
                        continue
                
                duration = (close_time - entry_time).total_seconds() / 60  # en minutes
                durations.append(duration)
        
        avg_duration = sum(durations) / len(durations) if durations else 0
        
        # Analyser les facteurs de succès
        success_factors = self._analyze_success_factors(successful_trades, failed_trades)
        
        # Calculer les performances par paire
        performance_by_pair = {}
        
        for position in filtered_positions:
            symbol = position.get("symbol", "UNKNOWN")
            pnl = position.get("pnl_percent", 0)
            
            if symbol not in performance_by_pair:
                performance_by_pair[symbol] = {
                    "count": 0,
                    "wins": 0,
                    "losses": 0,
                    "total_pnl": 0,
                    "avg_pnl": 0
                }
            
            performance_by_pair[symbol]["count"] += 1
            performance_by_pair[symbol]["total_pnl"] += pnl
            
            if pnl > 0:
                performance_by_pair[symbol]["wins"] += 1
            else:
                performance_by_pair[symbol]["losses"] += 1
        
        # Calculer les moyennes par paire
        for symbol in performance_by_pair:
            stats = performance_by_pair[symbol]
            stats["avg_pnl"] = stats["total_pnl"] / stats["count"] if stats["count"] > 0 else 0
            stats["win_rate"] = stats["wins"] / stats["count"] * 100 if stats["count"] > 0 else 0
        
        # Préparer le rapport
        report = {
            "success": True,
            "period": f"{days} jours",
            "total_trades": total_trades,
            "success_rate": success_rate,
            "avg_pnl": avg_pnl,
            "avg_win": avg_win,
            "avg_loss": avg_loss,
            "avg_duration_minutes": avg_duration,
            "success_factors": success_factors,
            "performance_by_pair": performance_by_pair,
            "timestamp": datetime.now().isoformat()
        }
        
        # Sauvegarder le rapport
        filename = os.path.join(self.analysis_dir, f"trade_analysis_{days}d_{datetime.now().strftime('%Y%m%d')}.json")
        
        try:
            with open(filename, 'w') as f:
                json.dump(report, f, indent=2, default=str)
            
            logger.info(f"Rapport d'analyse sauvegardé: {filename}")
        except Exception as e:
            logger.error(f"Erreur lors de la sauvegarde du rapport: {str(e)}")
        
        return report
    
    def _analyze_success_factors(self, successful_trades: List[Dict], failed_trades: List[Dict]) -> Dict:
        """
        Analyse les facteurs qui contribuent au succès ou à l'échec des trades
        
        Args:
            successful_trades: Liste des trades réussis
            failed_trades: Liste des trades échoués
            
        Returns:
            Analyse des facteurs de succès
        """
        # Extraire les facteurs des trades réussis et échoués
        success_factors = {}
        
        # Analyser les raisons de réussite
        if successful_trades:
            # Analyser le score moyen des trades réussis
            avg_score = sum(t.get("score", 0) for t in successful_trades) / len(successful_trades)
            success_factors["avg_score"] = avg_score
            
            # Analyser les signaux les plus fréquents dans les trades réussis
            signal_counts = {}
            for trade in successful_trades:
                signals = trade.get("signals", {}).get("signals", [])
                for signal in signals:
                    if signal not in signal_counts:
                        signal_counts[signal] = 0
                    signal_counts[signal] += 1
            
            # Trier les signaux par fréquence
            sorted_signals = sorted(signal_counts.items(), key=lambda x: x[1], reverse=True)
            success_factors["top_signals"] = [{"signal": s[0], "count": s[1]} for s in sorted_signals[:5]]
            
            # Analyser les conditions de marché
            market_conditions = {}
            for trade in successful_trades:
                conditions = trade.get("market_conditions", {}).get("details", {})
                for key, value in conditions.items():
                    if key not in market_conditions:
                        market_conditions[key] = []
                    market_conditions[key].append(value)
            
            # Calculer les moyennes des conditions de marché
            for key, values in market_conditions.items():
                if values and isinstance(values[0], (int, float)):
                    market_conditions[key] = sum(values) / len(values)
            
            success_factors["market_conditions"] = market_conditions
        
        # Analyser les raisons d'échec
        failure_factors = {}
        
        if failed_trades:
            # Analyser le score moyen des trades échoués
            avg_score = sum(t.get("score", 0) for t in failed_trades) / len(failed_trades)
            failure_factors["avg_score"] = avg_score
            
            # Analyser les signaux les plus fréquents dans les trades échoués
            signal_counts = {}
            for trade in failed_trades:
                signals = trade.get("signals", {}).get("signals", [])
                for signal in signals:
                    if signal not in signal_counts:
                        signal_counts[signal] = 0
                    signal_counts[signal] += 1
            
            # Trier les signaux par fréquence
            sorted_signals = sorted(signal_counts.items(), key=lambda x: x[1], reverse=True)
            failure_factors["top_signals"] = [{"signal": s[0], "count": s[1]} for s in sorted_signals[:5]]
        
        # Comparer les facteurs de réussite et d'échec
        comparison = {}
        
        if successful_trades and failed_trades:
            # Comparer les scores
            score_diff = success_factors.get("avg_score", 0) - failure_factors.get("avg_score", 0)
            comparison["score_difference"] = score_diff
            
            # Identifier les signaux qui discriminent le mieux les trades réussis des trades échoués
            success_signal_freq = {s["signal"]: s["count"] / len(successful_trades) for s in success_factors.get("top_signals", [])}
            failure_signal_freq = {s["signal"]: s["count"] / len(failed_trades) for s in failure_factors.get("top_signals", [])}
            
            discriminating_signals = []
            
            for signal, success_freq in success_signal_freq.items():
                failure_freq = failure_signal_freq.get(signal, 0)
                if success_freq > failure_freq:
                    discriminating_signals.append({
                        "signal": signal,
                        "success_freq": success_freq,
                        "failure_freq": failure_freq,
                        "difference": success_freq - failure_freq
                    })
            
            # Trier par différence de fréquence
            discriminating_signals = sorted(discriminating_signals, key=lambda x: x["difference"], reverse=True)
            comparison["discriminating_signals"] = discriminating_signals[:5]
        
        return {
            "success_factors": success_factors,
            "failure_factors": failure_factors,
            "comparison": comparison
        }
    
    def generate_recommendations(self) -> Dict:
        """
        Génère des recommandations pour améliorer la stratégie
        
        Returns:
            Dictionnaire avec les recommandations
        """
        # Analyser les trades récents
        analysis = self.analyze_recent_trades(days=30)
        
        if not analysis.get("success", False):
            return {
                "success": False,
                "message": "Impossible de générer des recommandations"
            }
        
        recommendations = {
            "success": True,
            "timestamp": datetime.now().isoformat(),
            "general_recommendations": [],
            "parameter_adjustments": [],
            "pair_recommendations": []
        }
        
        # Recommandations générales
        success_rate = analysis.get("success_rate", 0)
        avg_pnl = analysis.get("avg_pnl", 0)
        
        if success_rate < 50:
            recommendations["general_recommendations"].append({
                "importance": "high",
                "recommendation": "Augmenter le seuil de score minimum pour entrer en position",
                "reasoning": f"Taux de réussite faible ({success_rate:.1f}%)"
            })
        
        if avg_pnl < 0:
            recommendations["general_recommendations"].append({
                "importance": "high",
                "recommendation": "Réévaluer la stratégie de gestion des risques",
                "reasoning": f"P&L moyen négatif ({avg_pnl:.2f}%)"
            })
        
        # Recommandations par paire
        for symbol, stats in analysis.get("performance_by_pair", {}).items():
            if stats["count"] >= 5:  # Au moins 5 trades pour une analyse significative
                if stats["win_rate"] < 40:
                    recommendations["pair_recommendations"].append({
                        "pair": symbol,
                        "recommendation": "Éviter de trader cette paire temporairement",
                        "reasoning": f"Faible taux de réussite ({stats['win_rate']:.1f}%)"
                    })
                elif stats["win_rate"] > 70:
                    recommendations["pair_recommendations"].append({
                        "pair": symbol,
                        "recommendation": "Augmenter l'allocation sur cette paire",
                        "reasoning": f"Taux de réussite élevé ({stats['win_rate']:.1f}%)"
                    })
        
        # Recommandations sur les paramètres
        success_factors = analysis.get("success_factors", {}).get("comparison", {}).get("discriminating_signals", [])
        
        for factor in success_factors:
            signal = factor.get("signal", "")
            difference = factor.get("difference", 0)
            
            if difference > 0.3:  # Signal significativement plus fréquent dans les trades réussis
                if "RSI" in signal:
                    recommendations["parameter_adjustments"].append({
                        "parameter": "rsi_weight",
                        "recommendation": "Augmenter le poids du RSI dans le scoring",
                        "reasoning": f"Signal '{signal}' fortement associé aux trades réussis"
                    })
                elif "Bollinger" in signal:
                    recommendations["parameter_adjustments"].append({
                        "parameter": "bollinger_weight",
                        "recommendation": "Augmenter le poids des bandes de Bollinger dans le scoring",
                        "reasoning": f"Signal '{signal}' fortement associé aux trades réussis"
                    })
                elif "volume" in signal.lower():
                    recommendations["parameter_adjustments"].append({
                        "parameter": "volume_weight",
                        "recommendation": "Augmenter le poids des signaux de volume dans le scoring",
                        "reasoning": f"Signal '{signal}' fortement associé aux trades réussis"
                    })
        
        # Sauvegarder les recommandations
        filename = os.path.join(self.analysis_dir, f"recommendations_{datetime.now().strftime('%Y%m%d')}.json")
        
        try:
            with open(filename, 'w') as f:
                json.dump(recommendations, f, indent=2, default=str)
            
            logger.info(f"Recommandations sauvegardées: {filename}")
        except Exception as e:
            logger.error(f"Erreur lors de la sauvegarde des recommandations: {str(e)}")
        
        return recommendations
