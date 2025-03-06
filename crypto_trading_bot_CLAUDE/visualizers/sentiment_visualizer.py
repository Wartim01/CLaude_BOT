"""
Module de visualisation des données de sentiment pour mieux comprendre
l'impact des nouvelles sur les prix des cryptomonnaies
"""
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
from datetime import datetime, timedelta
from typing import Dict, List, Optional
import json

from ai.news_sentiment_analyzer import NewsSentimentAnalyzer
from config.config import DATA_DIR
from utils.logger import setup_logger

logger = setup_logger("sentiment_visualizer")

class SentimentVisualizer:
    """
    Visualise l'évolution du sentiment des nouvelles crypto et sa relation avec les prix
    """
    def __init__(self, sentiment_analyzer: Optional[NewsSentimentAnalyzer] = None):
        """
        Initialise le visualiseur de sentiment
        
        Args:
            sentiment_analyzer: Instance d'analyseur de sentiment (ou None pour en créer une nouvelle)
        """
        self.sentiment_analyzer = sentiment_analyzer or NewsSentimentAnalyzer()
        
        # Répertoire pour les graphiques
        self.output_dir = os.path.join(DATA_DIR, "visualizations", "sentiment")
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Style des visualisations
        self.setup_style()
    
    def setup_style(self):
        """Configure le style des visualisations"""
        sns.set_style("whitegrid")
        plt.rcParams.update({
            'font.size': 10,
            'axes.titlesize': 14,
            'axes.labelsize': 12,
            'xtick.labelsize': 10,
            'ytick.labelsize': 10,
            'legend.fontsize': 10,
            'figure.titlesize': 16,
            'figure.figsize': (12, 7)
        })
    
    def visualize_sentiment_history(self, symbol: str, days: int = 30, 
                                  with_price_data: Optional[pd.DataFrame] = None,
                                  save_path: Optional[str] = None) -> str:
        """
        Visualise l'historique du sentiment pour un symbole
        
        Args:
            symbol: Symbole crypto
            days: Nombre de jours d'historique
            with_price_data: DataFrame avec les données de prix (optionnel)
            save_path: Chemin pour sauvegarder le graphique (optionnel)
            
        Returns:
            Chemin du graphique sauvegardé
        """
        # Récupérer l'historique du sentiment
        sentiment_history = self.sentiment_analyzer.get_sentiment_history(symbol, days)
        
        if not sentiment_history:
            logger.warning(f"Pas d'historique de sentiment disponible pour {symbol}")
            return None
        
        # Préparer les données pour le graphique
        dates = [datetime.fromisoformat(entry["timestamp"]) for entry in sentiment_history]
        scores = [entry["sentiment_score"] for entry in sentiment_history]
        sentiments = [entry["dominant_sentiment"] for entry in sentiment_history]
        
        # Créer la figure
        fig, ax1 = plt.subplots(figsize=(12, 7))
        
        # Tracer le score de sentiment
        color = 'tab:blue'
        ax1.set_xlabel('Date')
        ax1.set_ylabel('Score de sentiment', color=color)
        sentiment_line = ax1.plot(dates, scores, 'o-', color=color, label="Sentiment")
        ax1.tick_params(axis='y', labelcolor=color)
        ax1.axhline(y=0, color='gray', linestyle='--', alpha=0.3)
        
        # Formater l'axe des dates
        ax1.xaxis.set_major_formatter(mdates.DateFormatter('%d/%m'))
        ax1.xaxis.set_major_locator(mdates.DayLocator(interval=max(1, days // 10)))
        
        # Ajouter les étiquettes des sentiments dominants
        for i, (date, score, sentiment) in enumerate(zip(dates, scores, sentiments)):
            if i % max(1, len(dates) // 10) == 0:  # Limiter le nombre d'étiquettes
                sentiment_color = 'green' if sentiment == 'bullish' else 'red' if sentiment == 'bearish' else 'gray'
                ax1.annotate(sentiment, (date, score), textcoords="offset points", 
                           xytext=(0, 10), ha='center', fontsize=8, color=sentiment_color)
        
        # Si des données de prix sont fournies, les ajouter sur un deuxième axe
        if with_price_data is not None and 'close' in with_price_data.columns:
            ax2 = ax1.twinx()
            color = 'tab:red'
            ax2.set_ylabel('Prix', color=color)
            
            # Aligner les données de prix avec les dates du sentiment
            start_date = min(dates) - timedelta(days=1)
            end_date = max(dates) + timedelta(days=1)
            
            price_df = with_price_data.loc[start_date:end_date]
            price_line = ax2.plot(price_df.index, price_df['close'], '-', color=color, alpha=0.7, label="Prix")
            ax2.tick_params(axis='y', labelcolor=color)
            
            # Ajouter les deux lignes à la légende
            lines = sentiment_line + price_line
            labels = [l.get_label() for l in lines]
            ax1.legend(lines, labels, loc='upper left')
        
        # Titre et légende
        plt.title(f"Évolution du sentiment pour {symbol} sur {days} jours")
        plt.grid(True, alpha=0.3)
        plt.xticks(rotation=45)
        plt.tight_layout()
        
        # Sauvegarder le graphique
        if save_path is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            save_path = os.path.join(self.output_dir, f"sentiment_history_{symbol}_{timestamp}.png")
        
        plt.savefig(save_path, dpi=300)
        plt.close()
        
        return save_path
    
    def visualize_sentiment_correlation(self, symbol: str, price_data: pd.DataFrame, 
                                      save_path: Optional[str] = None) -> str:
        """
        Visualise la corrélation entre le sentiment et les prix futurs
        
        Args:
            symbol: Symbole crypto
            price_data: DataFrame avec les données de prix
            save_path: Chemin pour sauvegarder le graphique (optionnel)
            
        Returns:
            Chemin du graphique sauvegardé
        """
        # Récupérer l'historique de sentiment
        sentiment_history = self.sentiment_analyzer.get_sentiment_history(symbol, days=90)
        
        if not sentiment_history or len(sentiment_history) < 5:
            logger.warning(f"Pas assez de données de sentiment pour {symbol}")
            return None
        
        if price_data is None or price_data.empty or 'close' not in price_data.columns:
            logger.warning("Données de prix invalides")
            return None
        
        # Créer un DataFrame à partir de l'historique de sentiment
        sentiment_df = pd.DataFrame([
            {
                "timestamp": datetime.fromisoformat(entry["timestamp"]),
                "sentiment_score": entry["sentiment_score"]
            }
            for entry in sentiment_history
        ])
        sentiment_df.set_index("timestamp", inplace=True)
        
        # Resampler les données de sentiment à la fréquence journalière
        daily_sentiment = sentiment_df.resample('1D').mean().dropna()
        
        # Préparer les données de prix
        price_df = price_data.copy()
        if not isinstance(price_df.index, pd.DatetimeIndex):
            price_df['timestamp'] = pd.to_datetime(price_df['timestamp'])
            price_df.set_index('timestamp', inplace=True)
        
        # Calculer les rendements futurs pour différentes périodes
        for days in [1, 3, 7]:
            price_df[f'return_{days}d'] = price_df['close'].pct_change(days).shift(-days)
        
        # Fusionner les données de sentiment et de prix
        merged_df = price_df.join(daily_sentiment, how='inner')
        merged_df.dropna(inplace=True)
        
        if len(merged_df) < 5:
            logger.warning(f"Pas assez de données alignées pour {symbol}")
            return None
        
        # Créer la visualisation de corrélation
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        
        time_periods = [1, 3, 7]
        for i, days in enumerate(time_periods):
            ax = axes[i]
            
            # Scatter plot
            sns.regplot(
                x='sentiment_score', 
                y=f'return_{days}d',
                data=merged_df,
                ax=ax,
                scatter_kws={'alpha': 0.5},
                line_kws={'color': 'red'}
            )
            
            # Calculer la corrélation
            corr = merged_df['sentiment_score'].corr(merged_df[f'return_{days}d'])
            
            # Titre et étiquettes
            ax.set_title(f"Corrélation à {days} jour(s): {corr:.3f}")
            ax.set_xlabel("Score de sentiment")
            ax.set_ylabel(f"Rendement à {days} jour(s)")
            ax.axhline(y=0, color='gray', linestyle='--', alpha=