# utils/visualizer.py
"""
Visualisation des performances et des trades
"""
import os
import json
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from typing import Dict, List, Optional, Union
from datetime import datetime, timedelta

from config.config import DATA_DIR
from utils.logger import setup_logger

logger = setup_logger("visualizer")

class TradeVisualizer:
    """
    Crée des visualisations pour les trades et les performances
    """
    def __init__(self, position_tracker):
        self.position_tracker = position_tracker
        self.output_dir = os.path.join(DATA_DIR, "visualizations")
        
        # Créer le répertoire de sortie s'il n'existe pas
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
    
    def plot_equity_curve(self, days: int = 30) -> str:
        """
        Génère une courbe d'équité sur la période spécifiée
        
        Args:
            days: Nombre de jours à inclure
            
        Returns:
            Chemin du fichier image généré
        """
        # Récupérer les positions fermées
        closed_positions = self.position_tracker.get_closed_positions(limit=1000)
        
        # Filtrer sur la période demandée
        start_date = datetime.now() - timedelta(days=days)
        filtered_positions = [
            p for p in closed_positions 
            if p.get("close_time") and p.get("close_time") > start_date
        ]
        
        if not filtered_positions:
            logger.warning(f"Aucune position fermée dans les {days} derniers jours")
            return ""
        
        # Trier par date de fermeture
        sorted_positions = sorted(
            filtered_positions,
            key=lambda p: p.get("close_time", datetime.min)
        )
        
        # Créer des listes pour le graphique
        dates = [p.get("close_time") for p in sorted_positions]
        pnls = [p.get("pnl_absolute", 0) for p in sorted_positions]
        
        # Calculer l'équité cumulée
        from config.config import INITIAL_CAPITAL
        equity = [INITIAL_CAPITAL]
        for pnl in pnls:
            equity.append(equity[-1] + pnl)
        
        equity = equity[1:]  # Supprimer le capital initial
        
        # Créer le graphique
        plt.figure(figsize=(12, 6))
        plt.plot(dates, equity, 'b-', linewidth=2)
        plt.title(f'Courbe d\'Équité sur les {days} derniers jours')
        plt.xlabel('Date')
        plt.ylabel('Équité (USDT)')
        plt.grid(True)
        
        # Formater l'axe des dates
        plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        plt.gcf().autofmt_xdate()
        
        # Ajouter des annotations
        initial_equity = INITIAL_CAPITAL
        final_equity = equity[-1]
        roi = (final_equity - initial_equity) / initial_equity * 100
        
        plt.annotate(f'ROI: {roi:.2f}%', 
                    xy=(0.02, 0.95), 
                    xycoords='axes fraction',
                    fontsize=12,
                    bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8))
        
        # Sauvegarder le graphique
        filename = os.path.join(self.output_dir, f'equity_curve_{days}d.png')
        plt.savefig(filename)
        plt.close()
        
        logger.info(f"Courbe d'équité générée: {filename}")
        return filename
    
    def plot_trade_analysis(self, days: int = 30) -> str:
        """
        Génère une analyse visuelle des trades sur la période spécifiée
        
        Args:
            days: Nombre de jours à inclure
            
        Returns:
            Chemin du fichier image généré
        """
        # Récupérer les positions fermées
        closed_positions = self.position_tracker.get_closed_positions(limit=1000)
        
        # Filtrer sur la période demandée
        start_date = datetime.now() - timedelta(days=days)
        filtered_positions = [
            p for p in closed_positions 
            if p.get("close_time") and p.get("close_time") > start_date
        ]
        
        if not filtered_positions:
            logger.warning(f"Aucune position fermée dans les {days} derniers jours")
            return ""
        
        # Créer une figure avec plusieurs sous-graphiques
        fig, axs = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle(f'Analyse des Trades sur les {days} derniers jours', fontsize=16)
        
        # 1. Distribution des profits/pertes
        pnls = [p.get("pnl_percent", 0) for p in filtered_positions]
        axs[0, 0].hist(pnls, bins=20, color='skyblue', edgecolor='black')
        axs[0, 0].set_title('Distribution des Profits/Pertes (%)')
        axs[0, 0].set_xlabel('Profit/Perte (%)')
        axs[0, 0].set_ylabel('Nombre de Trades')
        axs[0, 0].grid(True, alpha=0.3)
        
        # 2. Performance par paire de trading
        pairs = {}
        for p in filtered_positions:
            symbol = p.get("symbol", "UNKNOWN")
            pnl = p.get("pnl_absolute", 0)
            
            if symbol not in pairs:
                pairs[symbol] = {'count': 0, 'pnl': 0}
            
            pairs[symbol]['count'] += 1
            pairs[symbol]['pnl'] += pnl
        
        symbols = list(pairs.keys())
        pnls = [pairs[s]['pnl'] for s in symbols]
        
        # Tri par P&L
        sorted_indices = sorted(range(len(pnls)), key=lambda i: pnls[i])
        symbols = [symbols[i] for i in sorted_indices]
        pnls = [pnls[i] for i in sorted_indices]
        
        axs[0, 1].barh(symbols, pnls, color=['red' if p < 0 else 'green' for p in pnls])
        axs[0, 1].set_title('P&L par Paire de Trading (USDT)')
        axs[0, 1].set_xlabel('P&L (USDT)')
        axs[0, 1].grid(True, alpha=0.3)
        
        # 3. Ratio de réussite par jour de la semaine
        day_performance = {i: {'wins': 0, 'losses': 0} for i in range(7)}
        
        for p in filtered_positions:
            if p.get("close_time"):
                day = p.get("close_time").weekday()
                pnl = p.get("pnl_absolute", 0)
                
                if pnl >= 0:
                    day_performance[day]['wins'] += 1
                else:
                    day_performance[day]['losses'] += 1
        
        days = ['Lundi', 'Mardi', 'Mercredi', 'Jeudi', 'Vendredi', 'Samedi', 'Dimanche']
        win_rates = []
        
        for i in range(7):
            wins = day_performance[i]['wins']
            losses = day_performance[i]['losses']
            total = wins + losses
            
            if total > 0:
                win_rates.append(wins / total * 100)
            else:
                win_rates.append(0)
        
        axs[1, 0].bar(days, win_rates, color='orange')
        axs[1, 0].set_title('Ratio de Réussite par Jour de la Semaine')
        axs[1, 0].set_xlabel('Jour')
        axs[1, 0].set_ylabel('Ratio de Réussite (%)')
        axs[1, 0].set_ylim([0, 100])
        axs[1, 0].grid(True, alpha=0.3)
        
        # 4. Performance au fil du temps
        dates = [p.get("close_time") for p in filtered_positions]
        pnls = [p.get("pnl_absolute", 0) for p in filtered_positions]
        
        # Trier par date
        sorted_indices = sorted(range(len(dates)), key=lambda i: dates[i])
        dates = [dates[i] for i in sorted_indices]
        pnls = [pnls[i] for i in sorted_indices]
        
        # Calculer le cumul
        cumulative_pnl = [pnls[0]]
        for pnl in pnls[1:]:
            cumulative_pnl.append(cumulative_pnl[-1] + pnl)
        
        axs[1, 1].plot(dates, cumulative_pnl, 'b-')
        axs[1, 1].set_title('P&L Cumulatif au Fil du Temps')
        axs[1, 1].set_xlabel('Date')
        axs[1, 1].set_ylabel('P&L Cumulatif (USDT)')
        axs[1, 1].grid(True, alpha=0.3)
        axs[1, 1].xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        
        # Ajuster la mise en page
        plt.tight_layout(rect=[0, 0, 1, 0.95])
        
        # Sauvegarder le graphique
        filename = os.path.join(self.output_dir, f'trade_analysis_{days}d.png')
        plt.savefig(filename)
        plt.close()
        
        logger.info(f"Analyse des trades générée: {filename}")
        return filename
    
    def plot_trade_history(self, symbol: str, data_fetcher, position_id: str) -> str:
        """
        Génère un graphique montrant un trade spécifique avec entrée, sortie, et évolution du prix
        
        Args:
            symbol: Paire de trading
            data_fetcher: Récupérateur de données
            position_id: ID de la position
            
        Returns:
            Chemin du fichier image généré
        """
        # Récupérer les données de la position
        position = None
        
        # Rechercher d'abord dans les positions fermées
        for p in self.position_tracker.get_closed_positions():
            if p.get("id") == position_id:
                position = p
                break
        
        # Si non trouvée, rechercher dans les positions ouvertes
        if not position:
            position = self.position_tracker.get_position(position_id)
        
        if not position:
            logger.error(f"Position {position_id} non trouvée")
            return ""
        
        # Récupérer les données OHLCV
        from config.config import PRIMARY_TIMEFRAME
        
        # Déterminer la période à visualiser
        entry_time = position.get("entry_time")
        close_time = position.get("close_time")
        
        if not entry_time:
            logger.error(f"Heure d'entrée non disponible pour la position {position_id}")
            return ""
        
        # Si la position est toujours ouverte, utiliser l'heure actuelle
        if not close_time:
            close_time = datetime.now()
        
        # Ajouter une marge avant et après
        start_time = entry_time - timedelta(hours=2)
        end_time = close_time + timedelta(hours=2)
        
        # Convertir en millisecondes pour l'API
        start_ms = int(start_time.timestamp() * 1000)
        end_ms = int(end_time.timestamp() * 1000)
        
        # Récupérer les données
        ohlcv = data_fetcher.get_ohlcv(
            symbol, PRIMARY_TIMEFRAME, 
            start_time=start_ms, end_time=end_ms
        )
        
        if ohlcv.empty:
            logger.error(f"Données OHLCV non disponibles pour {symbol}")
            return ""
        
        # Créer le graphique
        plt.figure(figsize=(12, 6))
        
        # Graphique des prix
        plt.plot(ohlcv.index, ohlcv['close'], 'b-', linewidth=1.5)
        
        # Marquer l'entrée
        entry_price = position.get("entry_price")
        plt.axhline(y=entry_price, color='g', linestyle='--', alpha=0.5)
        plt.plot(entry_time, entry_price, 'go', markersize=8)
        
        # Marquer la sortie si la position est fermée
        if close_time and close_time != datetime.now():
            close_price = position.get("close_data", {}).get("fills", [{}])[0].get("price")
            if close_price:
                close_price = float(close_price)
                plt.plot(close_time, close_price, 'ro', markersize=8)
        
        # Marquer le stop-loss et le take-profit
        stop_loss = position.get("stop_loss_price")
        take_profit = position.get("take_profit_price")
        
        plt.axhline(y=stop_loss, color='r', linestyle='--', alpha=0.5)
        plt.axhline(y=take_profit, color='g', linestyle='--', alpha=0.5)
        
        # Ajouter des annotations
        side = position.get("side")
        pnl_percent = position.get("pnl_percent", 0)
        pnl_absolute = position.get("pnl_absolute", 0)
        
        title = f'Trade {position_id} - {symbol} ({side})'
        if pnl_percent != 0:
            title += f' - P&L: {pnl_percent:.2f}% ({pnl_absolute:.2f} USDT)'
        
        plt.title(title)
        plt.xlabel('Date')
        plt.ylabel('Prix')
        plt.grid(True)
        
        # Formater l'axe des dates
        plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d %H:%M'))
        plt.gcf().autofmt_xdate()
        
        # Ajouter une légende
        plt.legend(['Prix', 'Entrée', 'Sortie', 'Stop-Loss', 'Take-Profit'], 
                 loc='best')
        
        # Sauvegarder le graphique
        filename = os.path.join(self.output_dir, f'trade_{position_id}.png')
        plt.savefig(filename)
        plt.close()
        
        logger.info(f"Graphique du trade généré: {filename}")
        return filename