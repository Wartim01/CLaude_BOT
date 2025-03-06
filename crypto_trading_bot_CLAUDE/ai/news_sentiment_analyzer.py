"""
Module d'analyse du sentiment des nouvelles sur les cryptomonnaies
Récupère des articles de diverses sources et analyse leur sentiment
pour détecter les tendances du marché basées sur l'opinion publique
"""
import os
import json
import pandas as pd
import numpy as np
import requests
import re
import time
import datetime
from typing import Dict, List, Optional, Union, Any
from collections import defaultdict
from datetime import datetime, timedelta
import threading
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from newspaper import Article
import concurrent.futures

# Import internes
from config.config import DATA_DIR, API_KEYS
from utils.logger import setup_logger

# Configuration du logger
logger = setup_logger("news_sentiment_analyzer")

# Vérifier que les ressources NLTK sont disponibles ou les télécharger
try:
    nltk.data.find('vader_lexicon')
except LookupError:
    nltk.download('vader_lexicon', quiet=True)
    nltk.download('punkt', quiet=True)

class NewsSentimentAnalyzer:
    """
    Analyseur de sentiment des nouvelles crypto
    Récupère des données d'actualités de diverses sources et analyse leur sentiment
    """
    def __init__(self, refresh_interval: int = 3600, max_articles_per_symbol: int = 50):
        """
        Initialise l'analyseur de sentiment des nouvelles
        
        Args:
            refresh_interval: Intervalle de rafraîchissement des données en secondes
            max_articles_per_symbol: Nombre maximum d'articles à conserver par symbole
        """
        self.refresh_interval = refresh_interval
        self.max_articles_per_symbol = max_articles_per_symbol
        
        # Répertoire de données
        self.data_dir = os.path.join(DATA_DIR, "news_sentiment")
        os.makedirs(self.data_dir, exist_ok=True)
        
        # Données des articles et sentiments
        self.news_data = {}                    # Articles bruts
        self.sentiment_data = {}               # Sentiment analysé par symbole
        self.sentiment_history = defaultdict(list)  # Historique des scores de sentiment
        
        # Dictionnaire de correspondance nom <-> symbole
        self.crypto_mapping = {
            "bitcoin": ["BTC", "BTCUSDT", "BTCUSD"],
            "ethereum": ["ETH", "ETHUSDT", "ETHUSD"],
            "binance coin": ["BNB", "BNBUSDT", "BNBUSD"],
            "cardano": ["ADA", "ADAUSDT", "ADAUSD"],
            "solana": ["SOL", "SOLUSDT", "SOLUSD"],
            "ripple": ["XRP", "XRPUSDT", "XRPUSD"],
            "polkadot": ["DOT", "DOTUSDT", "DOTUSD"],
            "dogecoin": ["DOGE", "DOGEUSDT", "DOGEUSD"],
            # Ajouter d'autres mappings selon les besoins
        }
        
        # Inverser le mapping pour recherches rapides par symbole
        self.symbol_to_keywords = {}
        for keyword, symbols in self.crypto_mapping.items():
            for symbol in symbols:
                if symbol not in self.symbol_to_keywords:
                    self.symbol_to_keywords[symbol] = []
                self.symbol_to_keywords[symbol].append(keyword)
        
        # Analyseur de sentiment
        self.sentiment_analyzer = SentimentIntensityAnalyzer()
        
        # Thread de mise à jour automatique
        self.update_thread = None
        self.should_stop = False
        
        # Chargement des données précédentes
        self._load_cached_data()
        
    def start_auto_updates(self) -> bool:
        """
        Démarre les mises à jour automatiques en arrière-plan
        
        Returns:
            Succès de l'opération
        """
        if self.update_thread is None or not self.update_thread.is_alive():
            self.should_stop = False
            self.update_thread = threading.Thread(target=self._auto_update_worker)
            self.update_thread.daemon = True
            self.update_thread.start()
            logger.info(f"Mises à jour automatiques des news démarrées (intervalle: {self.refresh_interval}s)")
            return True
        return False
    
    def stop_auto_updates(self) -> bool:
        """
        Arrête les mises à jour automatiques
        
        Returns:
            Succès de l'opération
        """
        self.should_stop = True
        if self.update_thread and self.update_thread.is_alive():
            self.update_thread.join(timeout=2.0)
            logger.info("Mises à jour automatiques des news arrêtées")
            return True
        return False
    
    def _auto_update_worker(self) -> None:
        """Thread de travail pour les mises à jour automatiques"""
        while not self.should_stop:
            try:
                self.update_news_data()
                self.update_sentiment_analysis()
                time.sleep(self.refresh_interval)
            except Exception as e:
                logger.error(f"Erreur dans le thread de mise à jour automatique: {str(e)}")
                time.sleep(60)
    
    def update_news_data(self) -> Dict:
        """
        Met à jour les données d'actualités à partir de diverses sources
        
        Returns:
            Statistiques de la mise à jour
        """
        stats = {"new_articles": 0, "updated_symbols": set()}
        
        try:
            # 1. Récupérer des articles depuis Crypto News API
            crypto_news = self._fetch_crypto_news_api()
            stats["crypto_news_api_count"] = len(crypto_news)
            
            # 2. Récupérer des articles depuis Crypto Compare
            crypto_compare_news = self._fetch_cryptocompare_news()
            stats["crypto_compare_count"] = len(crypto_compare_news)
            
            # 3. Récupérer des articles depuis Crypto Panic
            crypto_panic_news = self._fetch_cryptopanic_news()
            stats["crypto_panic_count"] = len(crypto_panic_news)
            
            # 4. Fusionner toutes les sources d'actualités
            all_news = crypto_news + crypto_compare_news + crypto_panic_news
            
            # 5. Extraire le contenu des articles et associer aux symboles
            with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
                future_to_article = {executor.submit(self._process_article, article): article for article in all_news}
                for future in concurrent.futures.as_completed(future_to_article):
                    article = future_to_article[future]
                    try:
                        processed = future.result()
                        if processed:
                            stats["new_articles"] += 1
                            for symbol in processed["symbols"]:
                                stats["updated_symbols"].add(symbol)
                    except Exception as e:
                        logger.error(f"Erreur lors du traitement de l'article: {str(e)}")
            
            stats["updated_symbols"] = list(stats["updated_symbols"])
            stats["update_time"] = datetime.now().isoformat()
            
            logger.info(f"Données d'actualités mises à jour: {stats['new_articles']} nouveaux articles")
            
            # 6. Sauvegarder les données
            self._save_news_data()
            
            return stats
        
        except Exception as e:
            logger.error(f"Erreur lors de la mise à jour des données d'actualités: {str(e)}")
            stats["error"] = str(e)
            return stats
    
    def _process_article(self, article_data: Dict) -> Optional[Dict]:
        """
        Traite un article pour en extraire le contenu et l'associer aux symboles
        
        Args:
            article_data: Données brutes de l'article
            
        Returns:
            Article traité ou None en cas d'échec
        """
        try:
            # 1. Extraire les informations de base
            article_id = article_data.get("id") or article_data.get("guid") or str(hash(article_data.get("url", "") + str(time.time())))
            title = article_data.get("title", "")
            url = article_data.get("url", "")
            published_at = article_data.get("published_at") or article_data.get("publishedAt") or datetime.now().isoformat()
            
            # Si l'article existe déjà, le sauter
            if article_id in self.news_data:
                return None
            
            # 2. Télécharger et parser le contenu complet si disponible
            content = article_data.get("content", "")
            if not content and url:
                try:
                    article = Article(url)
                    article.download()
                    article.parse()
                    content = article.text
                except Exception as e:
                    logger.debug(f"Impossible de télécharger le contenu de {url}: {str(e)}")
                    content = article_data.get("summary", "") or article_data.get("body", "") or ""
            
            # 3. Identifier les symboles associés
            symbols = set()
            
            # Chercher dans les tags ou catégories si disponibles
            tags = article_data.get("tags", []) or article_data.get("categories", [])
            for tag in tags:
                tag_text = tag.lower() if isinstance(tag, str) else tag.get("name", "").lower()
                for keyword, symbol_list in self.crypto_mapping.items():
                    if keyword in tag_text:
                        symbols.update(symbol_list)
            
            # Chercher dans le titre et le contenu
            text_to_search = (title + " " + content).lower()
            for keyword, symbol_list in self.crypto_mapping.items():
                if keyword in text_to_search:
                    symbols.update(symbol_list)
            
            # Si aucun symbole spécifique trouvé, marquer comme "général"
            if not symbols:
                symbols = {"GENERAL"}
            
            # 4. Créer l'entrée d'article
            processed_article = {
                "id": article_id,
                "title": title,
                "url": url,
                "published_at": published_at,
                "symbols": list(symbols),
                "content": content[:10000],  # Limiter la taille du contenu
                "source": article_data.get("source", {}).get("name", "unknown")
            }
            
            # 5. Stocker l'article
            self.news_data[article_id] = processed_article
            
            # 6. Analyser le sentiment immédiatement
            sentiment_scores = self._analyze_text_sentiment(title + " " + content)
            processed_article["sentiment"] = sentiment_scores
            
            # 7. Associer aux symboles dans la structure de données
            for symbol in symbols:
                if symbol not in self.sentiment_data:
                    self.sentiment_data[symbol] = {
                        "articles": [],
                        "sentiment_scores": []
                    }
                
                # Ajouter l'ID de l'article à la liste des articles du symbole
                self.sentiment_data[symbol]["articles"].append(article_id)
                self.sentiment_data[symbol]["articles"] = self.sentiment_data[symbol]["articles"][-self.max_articles_per_symbol:]
                
                # Ajouter le score de sentiment
                self.sentiment_data[symbol]["sentiment_scores"].append({
                    "id": article_id,
                    "timestamp": datetime.now().isoformat(),
                    "score": sentiment_scores["compound"]
                })
                self.sentiment_data[symbol]["sentiment_scores"] = self.sentiment_data[symbol]["sentiment_scores"][-self.max_articles_per_symbol:]
            
            return processed_article
        
        except Exception as e:
            logger.error(f"Erreur lors du traitement de l'article: {str(e)}")
            return None
    
    def _fetch_crypto_news_api(self) -> List[Dict]:
        """
        Récupère des articles depuis Crypto News API
        
        Returns:
            Liste d'articles
        """
        # Ceci est un exemple avec une API fictive.
        # Remplacer par une API réelle ou utiliser un service gratuit comme News API
        api_key = API_KEYS.get("CRYPTO_NEWS_API")
        if not api_key:
            logger.warning("Clé API CRYPTO_NEWS_API non configurée")
            return []
        
        try:
            # Exemple de requête - à remplacer par l'URL réelle
            url = f"https://cryptonewsapi.example.com/api/v1/articles?apikey={api_key}"
            response = requests.get(url, timeout=10)
            
            if response.status_code == 200:
                data = response.json()
                return data.get("articles", [])
            else:
                logger.error(f"Erreur lors de la récupération des actualités: {response.status_code}")
                return []
                
        except Exception as e:
            logger.error(f"Erreur lors de l'appel à Crypto News API: {str(e)}")
            return []
    
    def _fetch_cryptocompare_news(self) -> List[Dict]:
        """
        Récupère des articles depuis CryptoCompare
        
        Returns:
            Liste d'articles
        """
        api_key = API_KEYS.get("CRYPTOCOMPARE")
        if not api_key:
            logger.warning("Clé API CRYPTOCOMPARE non configurée")
            return []
        
        try:
            url = "https://min-api.cryptocompare.com/data/v2/news/?lang=EN"
            headers = {"authorization": f"Apikey {api_key}"}
            response = requests.get(url, headers=headers, timeout=10)
            
            if response.status_code == 200:
                data = response.json()
                if data.get("Response") == "Success":
                    return data.get("Data", [])
                else:
                    logger.error(f"Erreur CryptoCompare: {data.get('Message')}")
                    return []
            else:
                logger.error(f"Erreur lors de la récupération des actualités: {response.status_code}")
                return []
                
        except Exception as e:
            logger.error(f"Erreur lors de l'appel à CryptoCompare: {str(e)}")
            return []
    
    def _fetch_cryptopanic_news(self) -> List[Dict]:
        """
        Récupère des articles depuis CryptoPanic
        
        Returns:
            Liste d'articles
        """
        api_key = API_KEYS.get("CRYPTOPANIC")
        if not api_key:
            logger.warning("Clé API CRYPTOPANIC non configurée")
            return []
        
        try:
            url = f"https://cryptopanic.com/api/v1/posts/?auth_token={api_key}&kind=news"
            response = requests.get(url, timeout=10)
            
            if response.status_code == 200:
                data = response.json()
                return data.get("results", [])
            else:
                logger.error(f"Erreur lors de la récupération des actualités: {response.status_code}")
                return []
                
        except Exception as e:
            logger.error(f"Erreur lors de l'appel à CryptoPanic: {str(e)}")
            return []
    
    def update_sentiment_analysis(self) -> Dict:
        """
        Met à jour l'analyse de sentiment pour tous les articles
        
        Returns:
            Résultats de l'analyse
        """
        results = {
            "total_articles_analyzed": 0,
            "symbols_updated": set(),
            "update_time": datetime.now().isoformat()
        }
        
        try:
            # 1. Analyser tous les articles qui n'ont pas encore de score de sentiment
            for article_id, article in self.news_data.items():
                if "sentiment" not in article:
                    text = article.get("title", "") + " " + article.get("content", "")
                    article["sentiment"] = self._analyze_text_sentiment(text)
                    results["total_articles_analyzed"] += 1
            
            # 2. Mettre à jour les scores agrégés par symbole
            for symbol in self.sentiment_data:
                self._update_symbol_sentiment(symbol)
                results["symbols_updated"].add(symbol)
            
            results["symbols_updated"] = list(results["symbols_updated"])
            
            # 3. Sauvegarder les données
            self._save_sentiment_data()
            
            logger.info(f"Analyse de sentiment mise à jour: {results['total_articles_analyzed']} articles")
            
            return results
        
        except Exception as e:
            logger.error(f"Erreur lors de la mise à jour de l'analyse de sentiment: {str(e)}")
            results["error"] = str(e)
            return results
    
    def _analyze_text_sentiment(self, text: str) -> Dict:
        """
        Analyse le sentiment d'un texte
        
        Args:
            text: Texte à analyser
            
        Returns:
            Scores de sentiment
        """
        if not text:
            return {"compound": 0, "pos": 0, "neu": 0, "neg": 0}
        
        try:
            # Nettoyer le texte
            text = text.lower()
            text = re.sub(r'http\S+', '', text)  # Supprimer les URLs
            text = re.sub(r'@\w+', '', text)     # Supprimer les mentions
            text = re.sub(r'#\w+', '', text)     # Supprimer les hashtags
            
            # Utilisation de VADER pour l'analyse de sentiment
            scores = self.sentiment_analyzer.polarity_scores(text)
            
            # Ajuster les scores avec des termes spécifiques aux cryptomonnaies
            crypto_pos_terms = ["bullish", "moon", "hodl", "buy the dip", "adoption", "breakout", "rally"]
            crypto_neg_terms = ["bearish", "crash", "dump", "sell off", "ban", "hack", "scam", "bubble"]
            
            for term in crypto_pos_terms:
                if term in text:
                    scores["compound"] = min(1.0, scores["compound"] + 0.05)
                    scores["pos"] = min(1.0, scores["pos"] + 0.05)
            
            for term in crypto_neg_terms:
                if term in text:
                    scores["compound"] = max(-1.0, scores["compound"] - 0.05)
                    scores["neg"] = min(1.0, scores["neg"] + 0.05)
            
            return scores
            
        except Exception as e:
            logger.error(f"Erreur lors de l'analyse de sentiment: {str(e)}")
            return {"compound": 0, "pos": 0, "neu": 0, "neg": 0}
    
    def _update_symbol_sentiment(self, symbol: str) -> None:
        """
        Met à jour l'agrégation de sentiment pour un symbole
        
        Args:
            symbol: Symbole à mettre à jour
        """
        if symbol not in self.sentiment_data:
            return
        
        # Récupérer les scores de sentiment récents
        recent_scores = []
        for score_info in self.sentiment_data[symbol]["sentiment_scores"]:
            try:
                # Filtrer pour ne prendre que les scores des 7 derniers jours
                score_time = datetime.fromisoformat(score_info["timestamp"])
                if datetime.now() - score_time <= timedelta(days=7):
                    recent_scores.append(score_info["score"])
            except:
                continue
        
        # Calculer les métriques de sentiment
        if recent_scores:
            avg_sentiment = np.mean(recent_scores)
            sentiment_std = np.std(recent_scores)
            num_bullish = sum(1 for score in recent_scores if score > 0.2)
            num_bearish = sum(1 for score in recent_scores if score < -0.2)
            num_neutral = len(recent_scores) - num_bullish - num_bearish
        else:
            avg_sentiment = 0
            sentiment_std = 0
            num_bullish = 0
            num_bearish = 0
            num_neutral = 0
        
        # Déterminer la tendance du sentiment
        trend = "stable"
        if len(recent_scores) >= 5:
            # Diviser en scores anciens et récents
            half = len(recent_scores) // 2
            older_scores = recent_scores[:half]
            newer_scores = recent_scores[half:]
            
            older_avg = np.mean(older_scores) if older_scores else 0
            newer_avg = np.mean(newer_scores) if newer_scores else 0
            
            if newer_avg > older_avg + 0.1:
                trend = "improving"
            elif newer_avg < older_avg - 0.1:
                trend = "deteriorating"
        
        # Déterminer le sentiment dominant
        dominant_sentiment = "neutral"
        if avg_sentiment > 0.2:
            dominant_sentiment = "bullish"
        elif avg_sentiment < -0.2:
            dominant_sentiment = "bearish"
        
        # Enregistrer le résultat de l'agrégation
        self.sentiment_data[symbol]["aggregated_sentiment"] = {
            "timestamp": datetime.now().isoformat(),
            "sentiment_score": float(avg_sentiment),
            "sentiment_std": float(sentiment_std),
            "dominant_sentiment": dominant_sentiment,
            "trend": trend,
            "bullish_count": num_bullish,
            "bearish_count": num_bearish,
            "neutral_count": num_neutral,
            "total_articles": len(recent_scores)
        }
        
        # Ajouter à l'historique
        self.sentiment_history[symbol].append({
            "timestamp": datetime.now().isoformat(),
            "sentiment_score": float(avg_sentiment),
            "dominant_sentiment": dominant_sentiment
        })
        
        # Limiter la taille de l'historique
        max_history = 100
        if len(self.sentiment_history[symbol]) > max_history:
            self.sentiment_history[symbol] = self.sentiment_history[symbol][-max_history:]
    
    def get_sentiment_for_symbol(self, symbol: str) -> Dict:
        """
        Récupère le sentiment pour un symbole spécifique
        
        Args:
            symbol: Symbole crypto (ex: "BTCUSDT")
            
        Returns:
            Données de sentiment pour le symbole
        """
        # Normaliser le symbole
        symbol = symbol.upper()
        
        # Si nous avons des données pour ce symbole
        if symbol in self.sentiment_data:
            aggregated = self.sentiment_data[symbol].get("aggregated_sentiment", {})
            
            # Vérifier si les données sont à jour
            if aggregated:
                # S'assurer que les données ne sont pas trop anciennes (< 24h)
                try:
                    last_update = datetime.fromisoformat(aggregated["timestamp"])
                    if datetime.now() - last_update > timedelta(hours=24):
                        self._update_symbol_sentiment(symbol)
                        aggregated = self.sentiment_data[symbol].get("aggregated_sentiment", {})
                except:
                    pass
                
                return aggregated
        
        # Si pas de données directes, chercher des noms alternatifs
        base_symbol = symbol.replace("USDT", "").replace("USD", "")
        for alt_symbol in [base_symbol, base_symbol + "USDT", base_symbol + "USD"]:
            if alt_symbol in self.sentiment_data:
                return self.sentiment_data[alt_symbol].get("aggregated_sentiment", {})
        
        # Si toujours pas de données, utiliser GENERAL comme fallback
        if "GENERAL" in self.sentiment_data:
            return self.sentiment_data["GENERAL"].get("aggregated_sentiment", {})
        
        # Aucune donnée disponible
        return {
            "timestamp": datetime.now().isoformat(),
            "sentiment_score": 0,
            "sentiment_std": 0,
            "dominant_sentiment": "neutral",
            "trend": "stable",
            "bullish_count": 0,
            "bearish_count": 0,
            "neutral_count": 0,
            "total_articles": 0,
            "error": "Pas de données disponibles pour ce symbole"
        }
    
    def get_sentiment_history(self, symbol: str, days: int = 30) -> List[Dict]:
        """
        Récupère l'historique du sentiment pour un symbole
        
        Args:
            symbol: Symbole crypto (ex: "BTCUSDT")
            days: Nombre de jours d'historique
            
        Returns:
            Liste des données de sentiment historiques
        """
        symbol = symbol.upper()
        
        # Calculer la date limite
        limit_date = datetime.now() - timedelta(days=days)
        
        # Rechercher le symbole et ses alternatives
        history = []
        symbols_to_check = [symbol]
        
        base_symbol = symbol.replace("USDT", "").replace("USD", "")
        symbols_to_check.extend([base_symbol, base_symbol + "USDT", base_symbol + "USD"])
        
        for sym in symbols_to_check:
            if sym in self.sentiment_history:
                for entry in self.sentiment_history[sym]:
                    try:
                        entry_date = datetime.fromisoformat(entry["timestamp"])
                        if entry_date > limit_date:
                            history.append(entry)
                    except:
                        continue
                
                # Si on a trouvé des données, ne pas chercher plus loin
                if history:
                    break
        
        # Trier par date
        history.sort(key=lambda x: x["timestamp"])
        
        return history
    
    def get_top_articles(self, symbol: str = None, limit: int = 10, min_sentiment: float = None) -> List[Dict]:
        """
        Récupère les articles les plus pertinents pour un symbole
        
        Args:
            symbol: Symbole crypto (optionnel, si None retourne les articles généraux)
            limit: Nombre maximum d'articles
            min_sentiment: Score de sentiment minimum (optionnel)
            
        Returns:
            Liste des articles pertinents
        """
        articles = []
        
        if symbol:
            symbol = symbol.upper()
            symbols_to_check = [symbol]
            
            base_symbol = symbol.replace("USDT", "").replace("USD", "")
            symbols_to_check.extend([base_symbol, base_symbol + "USDT", base_symbol + "USD"])
            
            for sym in symbols_to_check:
                if sym in self.sentiment_data:
                    # Récupérer les IDs des articles associés à ce symbole
                    article_ids = self.sentiment_data[sym].get("articles", [])
                    
                    # Récupérer les articles complets
                    for article_id in article_ids:
                        if article_id in self.news_data:
                            articles.append(self.news_data[article_id])
                    
                    # Si on a trouvé des articles, ne pas chercher plus loin
                    if articles:
                        break
        else:
            # Récupérer les articles généraux
            for article_id, article in self.news_data.items():
                if "GENERAL" in article.get("symbols", []):
                    articles.append(article)
        
        # Filtrer par sentiment si demandé
        if min_sentiment is not None:
            articles = [a for a in articles if a.get("sentiment", {}).get("compound", 0) >= min_sentiment]
        
        # Trier par date (du plus récent au plus ancien)
        articles.sort(key=lambda x: x.get("published_at", ""), reverse=True)
        
        # Limiter le nombre d'articles
        return articles[:limit]
    
    def _load_cached_data(self) -> None:
        """Charge les données depuis le disque"""
        try:
            news_file = os.path.join(self.data_dir, "news_data.json")
            sentiment_file = os.path.join(self.data_dir, "sentiment_data.json")
            history_file = os.path.join(self.data_dir, "sentiment_history.json")
            
            if os.path.exists(news_file):
                with open(news_file, 'r') as f:
                    self.news_data = json.load(f)
                logger.info(f"Données d'actualités chargées: {len(self.news_data)} articles")
            
            if os.path.exists(sentiment_file):
                with open(sentiment_file, 'r') as f:
                    self.sentiment_data = json.load(f)
                logger.info(f"Données de sentiment chargées: {len(self.sentiment_data)} symboles")
            
            if os.path.exists(history_file):
                with open(history_file, 'r') as f:
                    self.sentiment_history = defaultdict(list, json.load(f))
                logger.info(f"Historique de sentiment chargé: {len(self.sentiment_history)} symboles")
        except Exception as e:
            logger.error(f"Erreur lors du chargement des données: {str(e)}")
    
    def _save_news_data(self) -> None:
        """Sauvegarde les données d'actualités sur le disque"""
        try:
            news_file = os.path.join(self.data_dir, "news_data.json")
            with open(news_file, 'w') as f:
                json.dump(self.news_data, f, indent=2)
            logger.debug(f"Données d'actualités sauvegardées: {len(self.news_data)} articles")
        except Exception as e:
            logger.error(f"Erreur lors de la sauvegarde des données d'actualités: {str(e)}")
    
    def _save_sentiment_data(self) -> None:
        """Sauvegarde les données de sentiment sur le disque"""
        try:
            sentiment_file = os.path.join(self.data_dir, "sentiment_data.json")
            history_file = os.path.join(self.data_dir, "sentiment_history.json")
            
            with open(sentiment_file, 'w') as f:
                json.dump(self.sentiment_data, f, indent=2)
            
            with open(history_file, 'w') as f:
                json.dump(dict(self.sentiment_history), f, indent=2)
            
            logger.debug(f"Données de sentiment sauvegardées: {len(self.sentiment_data)} symboles")
        except Exception as e:
            logger.error(f"Erreur lors de la sauvegarde des données de sentiment: {str(e)}")
    
    def get_sentiment_summary(self, include_articles: bool = False) -> Dict:
        """
        Récupère un résumé global du sentiment du marché
        
        Args:
            include_articles: Inclure les articles récents dans le résumé
            
        Returns:
            Résumé du sentiment du marché
        """
        # Récupérer les sentiments pour les principales crypto
        main_cryptos = ["BTCUSDT", "ETHUSDT", "BNBUSDT", "SOLUSDT", "ADAUSDT"]
        crypto_sentiments = {}
        
        for symbol in main_cryptos:
            sentiment = self.get_sentiment_for_symbol(symbol)
            
            if sentiment:
                crypto_sentiments[symbol] = {
                    "score": sentiment.get("sentiment_score", 0),
                    "dominant": sentiment.get("dominant_sentiment", "neutral"),
                    "trend": sentiment.get("trend", "stable"),
                    "articles_count": sentiment.get("total_articles", 0)
                }
        
        # Calculer le sentiment global du marché
        sentiment_scores = [s.get("score", 0) for s in crypto_sentiments.values()]
        avg_sentiment = np.mean(sentiment_scores) if sentiment_scores else 0
        
        # Déterminer le sentiment dominant du marché
        market_sentiment = "neutral"
        if avg_sentiment > 0.2:
            market_sentiment = "bullish"
        elif avg_sentiment < -0.2:
            market_sentiment = "bearish"
        
        # Récupérer les articles récents si demandé
        recent_articles = []
        if include_articles:
            # Obtenir les articles les plus récents
            all_articles = []
            for article_id, article in self.news_data.items():
                all_articles.append({
                    "id": article_id,
                    "title": article.get("title", ""),
                    "url": article.get("url", ""),
                    "published_at": article.get("published_at", ""),
                    "sentiment": article.get("sentiment", {}).get("compound", 0),
                    "symbols": article.get("symbols", [])
                })
            
            # Trier par date (du plus récent au plus ancien)
            all_articles.sort(key=lambda x: x.get("published_at", ""), reverse=True)
            recent_articles = all_articles[:10]
        
        return {
            "timestamp": datetime.now().isoformat(),
            "market_sentiment": market_sentiment,
            "market_sentiment_score": float(avg_sentiment),
            "crypto_sentiments": crypto_sentiments,
            "recent_articles": recent_articles if include_articles else None,
            "data_sources": {
                "total_articles": len(self.news_data),
                "total_symbols": len(self.sentiment_data)
            }
        }
    
    def detect_sentiment_shifts(self, window_days: int = 3, threshold: float = 0.2) -> Dict:
        """
        Détecte les changements significatifs de sentiment pour tous les symboles
        
        Args:
            window_days: Fenêtre temporelle en jours pour la détection
            threshold: Seuil de changement pour considérer un shift significatif
            
        Returns:
            Dictionnaire des shifts de sentiment détectés par symbole
        """
        shifts = {}
        current_time = datetime.now()
        window_start = current_time - timedelta(days=window_days)
        
        # Vérifier chaque symbole pour les changements de sentiment
        for symbol, history in self.sentiment_history.items():
            if len(history) < 2:
                continue
            
            # Filtrer par la fenêtre temporelle
            recent_history = []
            for entry in history:
                try:
                    entry_time = datetime.fromisoformat(entry["timestamp"])
                    if entry_time >= window_start:
                        recent_history.append(entry)
                except:
                    continue
            
            if len(recent_history) < 2:
                continue
            
            # Trier par timestamp
            recent_history.sort(key=lambda x: x["timestamp"])
            
            # Comparer le sentiment le plus ancien au plus récent dans la fenêtre
            oldest = recent_history[0]
            newest = recent_history[-1]
            
            sentiment_shift = newest["sentiment_score"] - oldest["sentiment_score"]
            
            # Vérifier si le changement dépasse le seuil
            if abs(sentiment_shift) >= threshold:
                shift_type = "improving" if sentiment_shift > 0 else "deteriorating"
                
                shifts[symbol] = {
                    "shift": sentiment_shift,
                    "type": shift_type,
                    "from": {
                        "timestamp": oldest["timestamp"],
                        "score": oldest["sentiment_score"],
                        "sentiment": oldest["dominant_sentiment"]
                    },
                    "to": {
                        "timestamp": newest["timestamp"],
                        "score": newest["sentiment_score"],
                        "sentiment": newest["dominant_sentiment"]
                    },
                    "magnitude": abs(sentiment_shift)
                }
        
        # Trier par magnitude du changement
        sorted_shifts = {k: v for k, v in sorted(
            shifts.items(), 
            key=lambda item: item[1]["magnitude"], 
            reverse=True
        )}
        
        return {
            "shifts": sorted_shifts,
            "window_days": window_days,
            "threshold": threshold,
            "timestamp": current_time.isoformat(),
            "total_shifts_detected": len(sorted_shifts)
        }
    
    def analyze_sentiment_correlation(self, price_data: Dict[str, pd.DataFrame]) -> Dict:
        """
        Analyse la corrélation entre le sentiment et les mouvements de prix
        
        Args:
            price_data: Dictionnaire des DataFrames de prix par symbole
            
        Returns:
            Analyse des corrélations
        """
        results = {}
        
        for symbol, df in price_data.items():
            if symbol not in self.sentiment_history or len(self.sentiment_history[symbol]) < 5:
                continue
            
            # Préparation des données de sentiment
            sentiment_data = []
            for entry in self.sentiment_history[symbol]:
                try:
                    sentiment_data.append({
                        "timestamp": datetime.fromisoformat(entry["timestamp"]),
                        "score": entry["sentiment_score"]
                    })
                except:
                    continue
            
            if not sentiment_data:
                continue
            
            # Créer un DataFrame de sentiment
            sentiment_df = pd.DataFrame(sentiment_data)
            sentiment_df.set_index("timestamp", inplace=True)
            
            # Aligner les données de prix avec les données de sentiment
            aligned_data = pd.DataFrame()
            
            # Ajouter la colonne de prix (close)
            if "close" in df.columns:
                aligned_data["close"] = df["close"]
            
            # Resampler le sentiment à la même fréquence que les données de prix
            if not sentiment_df.empty and not aligned_data.empty:
                # Convertir l'index de sentiment_df en DateTimeIndex si nécessaire
                if not isinstance(sentiment_df.index, pd.DatetimeIndex):
                    sentiment_df.index = pd.to_datetime(sentiment_df.index)
                
                # Resampler et interpoler
                resampled_sentiment = sentiment_df.resample('1D').mean().interpolate(method='linear')
                
                # Fusionner avec les données de prix
                aligned_data = aligned_data.join(resampled_sentiment, how='left')
                aligned_data["score"].fillna(method='ffill', inplace=True)
                
                # Calculer les rendements futurs pour différentes périodes
                for days in [1, 3, 7]:
                    aligned_data[f"return_{days}d"] = aligned_data["close"].pct_change(days).shift(-days)
                
                # Supprimer les lignes avec des NaN
                aligned_data.dropna(inplace=True)
                
                # Calculer les corrélations
                if len(aligned_data) > 5:
                    correlations = {}
                    
                    for days in [1, 3, 7]:
                        corr = aligned_data["score"].corr(aligned_data[f"return_{days}d"])
                        correlations[f"{days}d"] = float(corr)
                    
                    # Évaluer l'utilité prédictive
                    best_corr = max([abs(c) for c in correlations.values()]) if correlations else 0
                    predictive_value = "high" if best_corr > 0.5 else "medium" if best_corr > 0.3 else "low"
                    
                    results[symbol] = {
                        "correlations": correlations,
                        "data_points": len(aligned_data),
                        "predictive_value": predictive_value,
                        "best_correlation": best_corr
                    }
        
        return {
            "timestamp": datetime.now().isoformat(),
            "results": results,
            "symbols_analyzed": len(results)
        }
