"""
Module qui surveille et évalue le risque global du marché des cryptomonnaies
pour ajuster dynamiquement les stratégies de trading
"""
import os
import json
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Union, Any
from datetime import datetime, timedelta
import threading
import time
import requests
from collections import deque

from config.config import DATA_DIR, API_KEYS
from utils.logger import setup_logger

logger = setup_logger("market_risk_feed")

class MarketRiskFeed:
    """
    Surveille et évalue les facteurs de risque du marché des cryptomonnaies
    pour aider à la prise de décision de trading
    """
    def __init__(self, refresh_interval: int = 3600):
        """
        Initialise le flux de risque du marché
        
        Args:
            refresh_interval: Intervalle de rafraîchissement des données en secondes
        """
        self.refresh_interval = refresh_interval
        
        # Répertoire de données
        self.data_dir = os.path.join(DATA_DIR, "market_risk")
        os.makedirs(self.data_dir, exist_ok=True)
        
        # Fichiers de données
        self.risk_file = os.path.join(self.data_dir, "market_risk_data.json")
        self.volatility_file = os.path.join(self.data_dir, "volatility_history.json")
        self.correlation_file = os.path.join(self.data_dir, "correlation_history.json")
        
        # Données en cache
        self.risk_data = {}
        self.volatility_history = {}
        self.correlation_history = {}
        
        # Différents niveaux de risque de marché
        self.risk_levels = {
            "low": (0, 30),       # 0-30% = risque faible
            "medium": (30, 60),   # 30-60% = risque moyen
            "high": (60, 80),     # 60-80% = risque élevé
            "extreme": (80, 100)  # 80-100% = risque extrême
        }
        
        # Histoire récente pour calculs
        self.recent_btc_volatility = deque(maxlen=30)  # 30 jours
        self.recent_market_cap = deque(maxlen=30)      # 30 jours
        self.recent_alts_correlation = deque(maxlen=30) # 30 jours
        
        # Thread pour mise à jour automatique
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
            logger.info(f"Mises à jour automatiques du risque de marché démarrées (intervalle: {self.refresh_interval}s)")
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
            logger.info("Mises à jour automatiques du risque de marché arrêtées")
            return True
        return False
    
    def _auto_update_worker(self) -> None:
        """Thread de travail pour les mises à jour automatiques"""
        while not self.should_stop:
            try:
                self.update_market_risk()
                time.sleep(self.refresh_interval)
            except Exception as e:
                logger.error(f"Erreur dans le thread de mise à jour automatique: {str(e)}")
                time.sleep(60)
    
    def update_market_risk(self) -> None:
        """
        Met à jour les données de risque de marché
        """
        try:
            # 1. Mettre à jour la volatilité du BTC
            self._update_btc_volatility()
            
            # 2. Mettre à jour la capitalisation du marché
            self._update_market_cap()
            
            # 3. Mettre à jour la corrélation des altcoins
            self._update_alts_correlation()
            
            # 4. Calculer le score de risque global
            self._calculate_global_risk()
            
            # 5. Sauvegarder les données en cache
            self._save_cached_data()
            
            logger.info("Données de risque de marché mises à jour avec succès")
        except Exception as e:
            logger.error(f"Erreur lors de la mise à jour des données de risque de marché: {str(e)}")
    
    def _update_btc_volatility(self) -> None:
        """
        Met à jour la volatilité du BTC
        """
        # Exemple de mise à jour de la volatilité du BTC
        # Vous pouvez utiliser des API externes pour obtenir les données réelles
        btc_volatility = np.random.uniform(0, 100)  # Valeur aléatoire pour l'exemple
        self.recent_btc_volatility.append(btc_volatility)
        self.volatility_history[datetime.now().isoformat()] = btc_volatility
        logger.debug(f"Volatilité du BTC mise à jour: {btc_volatility:.2f}%")
    
    def _update_market_cap(self) -> None:
        """
        Met à jour la capitalisation du marché
        """
        # Exemple de mise à jour de la capitalisation du marché
        # Vous pouvez utiliser des API externes pour obtenir les données réelles
        market_cap = np.random.uniform(1e9, 1e12)  # Valeur aléatoire pour l'exemple
        self.recent_market_cap.append(market_cap)
        self.risk_data["market_cap"] = market_cap
        logger.debug(f"Capitalisation du marché mise à jour: {market_cap:.2f} USD")
    
    def _update_alts_correlation(self) -> None:
        """
        Met à jour la corrélation des altcoins
        """
        # Exemple de mise à jour de la corrélation des altcoins
        # Vous pouvez utiliser des API externes pour obtenir les données réelles
        alts_correlation = np.random.uniform(-1, 1)  # Valeur aléatoire pour l'exemple
        self.recent_alts_correlation.append(alts_correlation)
        self.correlation_history[datetime.now().isoformat()] = alts_correlation
        logger.debug(f"Corrélation des altcoins mise à jour: {alts_correlation:.2f}")
    
    def _calculate_global_risk(self) -> None:
        """
        Calcule le score de risque global
        """
        # Exemple de calcul du score de risque global
        # Vous pouvez ajuster cette logique en fonction de vos besoins
        avg_btc_volatility = np.mean(self.recent_btc_volatility) if self.recent_btc_volatility else 0
        avg_market_cap = np.mean(self.recent_market_cap) if self.recent_market_cap else 0
        avg_alts_correlation = np.mean(self.recent_alts_correlation) if self.recent_alts_correlation else 0
        
        # Calcul du score de risque global (exemple simplifié)
        global_risk_score = (avg_btc_volatility + (1 - avg_market_cap / 1e12) * 100 + (1 - avg_alts_correlation) * 50) / 3
        self.risk_data["global_risk_score"] = global_risk_score
        
        # Détermination du niveau de risque
        for level, (min_score, max_score) in self.risk_levels.items():
            if min_score <= global_risk_score < max_score:
                self.risk_data["global_risk_level"] = level
                break
        
        logger.info(f"Score de risque global calculé: {global_risk_score:.2f} ({self.risk_data['global_risk_level']})")
    
    def _load_cached_data(self) -> None:
        """
        Charge les données en cache depuis le disque
        """
        try:
            if os.path.exists(self.risk_file):
                with open(self.risk_file, 'r') as f:
                    self.risk_data = json.load(f)
            if os.path.exists(self.volatility_file):
                with open(self.volatility_file, 'r') as f:
                    self.volatility_history = json.load(f)
            if os.path.exists(self.correlation_file):
                with open(self.correlation_file, 'r') as f:
                    self.correlation_history = json.load(f)
            logger.info("Données en cache chargées avec succès")
        except Exception as e:
            logger.error(f"Erreur lors du chargement des données en cache: {str(e)}")
    
    def _save_cached_data(self) -> None:
        """
        Sauvegarde les données en cache sur le disque
        """
        try:
            with open(self.risk_file, 'w') as f:
                json.dump(self.risk_data, f, indent=2)
            with open(self.volatility_file, 'w') as f:
                json.dump(self.volatility_history, f, indent=2)
            with open(self.correlation_file, 'w') as f:
                json.dump(self.correlation_history, f, indent=2)
            
            logger.debug("Données en cache sauvegardées avec succès")
        except Exception as e:
            logger.error(f"Erreur lors de la sauvegarde des données en cache: {str(e)}")
    
    def get_market_risk(self, symbol: str = None) -> Dict:
        """
        Récupère les données de risque du marché
        
        Args:
            symbol: Symbole pour lequel obtenir le risque spécifique (optionnel)
            
        Returns:
            Données de risque du marché
        """
        # Vérifier si nous avons besoin de mettre à jour les données
        if not self.risk_data or "last_update" not in self.risk_data:
            self.update_market_risk()
        elif "last_update" in self.risk_data:
            last_update = datetime.fromisoformat(self.risk_data["last_update"])
            if (datetime.now() - last_update) > timedelta(hours=1):
                self.update_market_risk()
        
        # Si un symbole est spécifié, ajuster le risque en fonction des spécificités du symbole
        if symbol:
            return self._adjust_risk_for_symbol(symbol)
        
        # Sinon, retourner le risque global
        return {
            "level": self.risk_data.get("global_risk_level", "medium"),
            "risk_score": self.risk_data.get("global_risk_score", 50.0),
            "btc_volatility": np.mean(self.recent_btc_volatility) if self.recent_btc_volatility else None,
            "market_cap_trend": "stable",
            "risk_factors": self._get_risk_factors(),
            "last_update": self.risk_data.get("last_update", datetime.now().isoformat()),
            "market_conditions": {
                "trend": self.risk_data.get("market_trend", "neutral"),
                "liquidity": self.risk_data.get("liquidity", "normal"),
                "sentiment": self.risk_data.get("sentiment", "neutral")
            }
        }
    
    def _adjust_risk_for_symbol(self, symbol: str) -> Dict:
        """
        Ajuste le risque global en fonction des spécificités d'un symbole
        
        Args:
            symbol: Symbole à analyser
            
        Returns:
            Risque ajusté pour le symbole
        """
        # Par défaut, utiliser le risque global
        base_risk = self.get_market_risk(None)
        risk_score = base_risk["risk_score"]
        
        # Ajuster le score pour les symboles spécifiques
        if symbol == "BTCUSDT":
            # Bitcoin est généralement moins risqué que le marché global
            risk_score = max(0, risk_score * 0.9)
        elif symbol in ["ETHUSDT", "BNBUSDT"]:
            # Les grandes altcoins ont un risque légèrement supérieur
            risk_score = min(100, risk_score * 1.1)
        elif any(high_risk in symbol for high_risk in ["SHIB", "DOGE", "MEME"]):
            # Les memecoins sont beaucoup plus risquées
            risk_score = min(100, risk_score * 1.3)
        
        # Déterminer le niveau de risque ajusté
        risk_level = "medium"
        for level, (min_score, max_score) in self.risk_levels.items():
            if min_score <= risk_score < max_score:
                risk_level = level
                break
        
        # Retourner le risque ajusté
        return {
            "level": risk_level,
            "risk_score": risk_score,
            "symbol": symbol,
            "global_risk_level": base_risk["level"],
            "symbol_specific_factors": self._get_symbol_specific_factors(symbol),
            "risk_factors": self._get_risk_factors(),
            "last_update": datetime.now().isoformat()
        }
    
    def _get_risk_factors(self) -> List[Dict]:
        """
        Récupère les facteurs de risque actuels
        
        Returns:
            Liste des facteurs de risque
        """
        risk_factors = []
        
        # Vérifier la volatilité du BTC
        if self.recent_btc_volatility and len(self.recent_btc_volatility) > 0:
            btc_vol = self.recent_btc_volatility[-1]
            if btc_vol > 80:
                risk_factors.append({
                    "factor": "btc_volatility",
                    "level": "extreme",
                    "description": "Volatilité extrême du Bitcoin"
                })
            elif btc_vol > 60:
                risk_factors.append({
                    "factor": "btc_volatility",
                    "level": "high",
                    "description": "Volatilité élevée du Bitcoin"
                })
        
        # Vérifier la capitalisation du marché
        if self.recent_market_cap and len(self.recent_market_cap) > 1:
            current_cap = self.recent_market_cap[-1]
            prev_cap = self.recent_market_cap[-2]
            change_pct = (current_cap - prev_cap) / prev_cap * 100
            
            if change_pct < -10:
                risk_factors.append({
                    "factor": "market_cap_decline",
                    "level": "high",
                    "description": f"Forte baisse de la capitalisation du marché ({change_pct:.1f}%)"
                })
        
        # Vérifier la corrélation des altcoins
        if self.recent_alts_correlation and len(self.recent_alts_correlation) > 0:
            correlation = self.recent_alts_correlation[-1]
            if correlation > 0.9:
                risk_factors.append({
                    "factor": "high_correlation",
                    "level": "medium",
                    "description": "Forte corrélation entre les altcoins indiquant un possible mouvement de marché"
                })
        
        return risk_factors
    
    def _get_symbol_specific_factors(self, symbol: str) -> List[Dict]:
        """
        Récupère les facteurs de risque spécifiques à un symbole
        
        Args:
            symbol: Symbole à analyser
            
        Returns:
            Liste des facteurs de risque spécifiques
        """
        # Ici, vous pourriez intégrer des données spécifiques aux symboles
        # comme la volatilité historique, l'activité récente, etc.
        
        # Pour l'exemple, nous retournons des facteurs génériques
        factors = []
        
        # Différencier les types de symboles
        if "BTC" in symbol:
            factors.append({
                "factor": "market_leader",
                "level": "low",
                "description": "Bitcoin tend à être moins volatile que les altcoins"
            })
        elif any(meme in symbol for meme in ["SHIB", "DOGE", "ELON", "PEPE"]):
            factors.append({
                "factor": "memecoin",
                "level": "high",
                "description": "Les memecoins sont très volatiles et sensibles au sentiment du marché"
            })
        elif any(defi in symbol for defi in ["UNI", "AAVE", "COMP", "MKR"]):
            factors.append({
                "factor": "defi_exposure",
                "level": "medium",
                "description": "Les tokens DeFi peuvent être sensibles aux failles de sécurité et aux changements réglementaires"
            })
        
        return factors
    
    def get_historical_risk(self, days: int = 30) -> Dict:
        """
        Récupère l'historique du risque de marché
        
        Args:
            days: Nombre de jours d'historique à récupérer
            
        Returns:
            Historique du risque de marché
        """
        # Pour une implémentation complète, vous auriez besoin de stocker l'historique des scores de risque
        # Ici, nous simulons un historique avec des données aléatoires
        
        start_date = datetime.now() - timedelta(days=days)
        dates = [start_date + timedelta(days=i) for i in range(days)]
        
        # Générer un historique fictif de scores de risque
        risk_scores = []
        risk_levels = []
        current_score = 50  # Score de risque initial moyen
        
        for _ in range(days):
            # Ajouter une variation aléatoire au score précédent
            change = np.random.normal(0, 5)  # Variation normale avec écart-type de 5
            current_score = max(0, min(100, current_score + change))  # Limiter entre 0 et 100
            risk_scores.append(current_score)
            
            # Déterminer le niveau de risque
            level = "medium"
            for lvl, (min_score, max_score) in self.risk_levels.items():
                if min_score <= current_score < max_score:
                    level = lvl
                    break
            risk_levels.append(level)
        
        # Formater l'historique
        history = []
        for i, date in enumerate(dates):
            history.append({
                "date": date.isoformat(),
                "risk_score": float(risk_scores[i]),
                "risk_level": risk_levels[i]
            })
        
        return {
            "history": history,
            "days": days,
            "start_date": start_date.isoformat(),
            "end_date": datetime.now().isoformat()
        }