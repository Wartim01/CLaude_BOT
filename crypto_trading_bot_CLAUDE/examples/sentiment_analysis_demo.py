"""
Démonstration de l'analyse de sentiment des nouvelles crypto
et son intégration dans le système de trading
"""
import os
import sys
import time
import pandas as pd
from datetime import datetime, timedelta

# Ajouter le répertoire parent au path pour les imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ai.news_sentiment_analyzer import NewsSentimentAnalyzer
from exchanges.binance_client import BinanceClient
from config.config import API_KEYS
from utils.logger import setup_logger

logger = setup_logger("sentiment_demo")

def main():
    print("Démarrage de la démonstration d'analyse de sentiment")
    
    # Initialiser l'analyseur de sentiment
    sentiment_analyzer = NewsSentimentAnalyzer(refresh_interval=1800)  # Rafraîchir toutes les 30 minutes
    
    # Charger les données initiales et démarrer les mises à jour automatiques
    print("Chargement des données d'actualités et analyse du sentiment...")
    sentiment_analyzer.update_news_data()
    sentiment_analyzer.update_sentiment_analysis()
    sentiment_analyzer.start_auto_updates()
    
    # Initialiser le client Binance pour récupérer les prix
    binance = BinanceClient(API_KEYS["BINANCE"]["key"], API_KEYS["BINANCE"]["secret"])
    
    # Liste des symboles à surveiller
    symbols = ["BTCUSDT", "ETHUSDT", "BNBUSDT", "ADAUSDT", "SOLUSDT", "DOGEUSDT"]
    
    try:
        # 1. Récupérer le sentiment pour chaque symbole
        print("\n--- Sentiment actuel par symbole ---")
        for symbol in symbols:
            sentiment = sentiment_analyzer.get_sentiment_for_symbol(symbol)
            
            # Formater le score de sentiment
            score = sentiment.get("sentiment_score", 0)
            score_str = f"{score:.2f}"
            if score > 0:
                score_str = f"+{score_str}"
            
            print(f"{symbol}: {sentiment.get('dominant_sentiment', 'neutral').upper()} ({score_str}) | " 
                  f"Tendance: {sentiment.get('trend', 'stable')} | "
                  f"Articles: {sentiment.get('total_articles', 0)}")
        
        # 2. Récupérer les articles récents pour BTC
        print("\n--- Articles récents pour Bitcoin ---")
        btc_articles = sentiment_analyzer.get_top_articles("BTCUSDT", limit=5)
        
        for i, article in enumerate(btc_articles, 1):
            sentiment_score = article.get("sentiment", {}).get("compound", 0)
            sentiment_str = "🔴" if sentiment_score < -0.1 else "🟢" if sentiment_score > 0.1 else "⚪"
            print(f"{i}. {sentiment_str} {article.get('title')} ({sentiment_score:.2f})")
            print(f"   Source: {article.get('source')} | URL: {article.get('url')}")
        
        # 3. Analyser les shifts de sentiment récents
        print("\n--- Changements significatifs de sentiment (3 derniers jours) ---")
        shifts = sentiment_analyzer.detect_sentiment_shifts(window_days=3, threshold=0.15)
        
        if shifts["total_shifts_detected"] > 0:
            for symbol, shift_data in list(shifts["shifts"].items())[:5]:  # Top 5
                direction = "↗️" if shift_data["type"] == "improving" else "↘️"
                print(f"{symbol}: {direction} {shift_data['from']['sentiment']} → {shift_data['to']['sentiment']} "
                      f"(Δ: {shift_data['shift']:.2f})")
        else:
            print("Aucun changement significatif de sentiment détecté.")
        
        # 4. Récupérer les prix et analyser les corrélations
        print("\n--- Analyse des corrélations sentiment-prix ---")
        
        # Récupérer les données de prix historiques
        price_data = {}
        for symbol in symbols:
            klines = binance.get_historical_klines(symbol, "1d", "21 days ago UTC")
            if klines:
                df = pd.DataFrame(klines, columns=["timestamp", "open", "high", "low", "close", "volume", 
                                                "close_time", "quote_volume", "trades", "taker_buy_base", 
                                                "taker_buy_quote", "ignored"])
                
                # Convertir les types
                for col in ["open", "high", "low", "close", "volume"]:
                    df[col] = df[col].astype(float)
                
                df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
                df.set_index("timestamp", inplace=True)
                
                price_data[symbol] = df
        
        # Analyser les corrélations
        correlations = sentiment_analyzer.analyze_sentiment_correlation(price_data)
        
        if correlations["symbols_analyzed"] > 0:
            for symbol, result in correlations["results"].items():
                print(f"{symbol}: Valeur prédictive {result['predictive_value'].upper()} | "
                      f"Meilleure corrélation: {result['best_correlation']:.2f}")
                
                # Détailler les corrélations par période
                corrs = result["correlations"]
                print(f"  ├─ 1 jour: {corrs.get('1d', 0):.2f}")
                print(f"  ├─ 3 jours: {corrs.get('3d', 0):.2f}")
                print(f"  └─ 7 jours: {corrs.get('7d', 0):.2f}")
        else:
            print("Pas assez de données pour l'analyse de corrélation.")
        
        # 5. Résumé global du marché
        print("\n--- Résumé global du sentiment du marché ---")
        market_summary = sentiment_analyzer.get_sentiment_summary()
        
        market_sentiment = market_summary["market_sentiment"].upper()
        sentiment_score = market_summary["market_sentiment_score"]
        
        print(f"Sentiment global: {market_sentiment} ({sentiment_score:.2f})")
        print(f"Basé sur {market_summary['data_sources']['total_articles']} articles pour {market_summary['data_sources']['total_symbols']} symboles")
        
        # Attendre que l'utilisateur quitte
        print("\nAppuyez sur Ctrl+C pour quitter...")
        while True:
            time.sleep(1)
            
    except KeyboardInterrupt:
        print("\nArrêt de la démonstration...")
    finally:
        # Arrêter les mises à jour automatiques
        sentiment_analyzer.stop_auto_updates()
        print("Démonstration terminée.")

if __name__ == "__main__":
    main()
