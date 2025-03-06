# strategies/market_state.py
"""
Module d'analyse de l'état du marché pour adapter les stratégies de trading
"""
import logging
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Union
from datetime import datetime, timedelta

from core.data_fetcher import MarketDataFetcher
from utils.logger import setup_logger
from indicators.trend import detect_trend
from indicators.volatility import calculate_volatility
from utils.network_utils import retry_with_backoff

logger = setup_logger("market_state")

class MarketStateAnalyzer:
    """
    Analyse l'état du marché pour adapter les stratégies de trading
    """
    def __init__(self, data_fetcher: MarketDataFetcher):
        self.data_fetcher = data_fetcher
        self.market_states = {}  # Cache des états de marché
        self.market_regimes = {}  # Classification des régimes de marché
        self.global_sentiment = "neutral"  # Sentiment global du marché
        
        # Fenêtres de temps pour l'analyse
        self.windows = {
            "short": 24,   # 1 jour (24 bougies de 1h)
            "medium": 72,  # 3 jours
            "long": 168    # 1 semaine
        }
        
        # Seuils pour la classification des états
        self.thresholds = {
            "volatility": {
                "low": 1.0,    # Volatilité faible: < 1%
                "high": 3.0    # Volatilité élevée: > 3%
            },
            "volume": {
                "low": 0.7,    # Volume faible: < 70% de la moyenne
                "high": 1.5    # Volume élevé: > 150% de la moyenne
            },
            "trend": {
                "weak": 0.3,   # Tendance faible: ADX < 30%
                "strong": 0.5  # Tendance forte: ADX > 50%
            }
        }
        
        logger.info("Analyseur d'état du marché initialisé")
    
    @retry_with_backoff(max_retries=2, base_delay=1.0)
    def analyze_market_state(self, symbol: str, timeframe: str = "1h") -> Dict:
        """
        Analyse complète de l'état du marché pour un symbole et un timeframe donnés
        
        Args:
            symbol: Paire de trading
            timeframe: Intervalle de temps pour l'analyse
            
        Returns:
            Dictionnaire avec l'état du marché
        """
        # Vérifier si nous avons un état récent en cache
        cache_key = f"{symbol}_{timeframe}"
        if cache_key in self.market_states:
            timestamp, state = self.market_states[cache_key]
            # Utiliser le cache si l'état a été calculé il y a moins de 15 minutes
            if (datetime.now() - timestamp).total_seconds() < 900:
                return state
        
        try:
            # Récupérer les données de marché
            ohlcv = self.data_fetcher.get_historical_data(symbol, timeframe, limit=200)
            if ohlcv is None or len(ohlcv) < 100:
                logger.warning(f"Données insuffisantes pour l'analyse du marché: {symbol} ({timeframe})")
                return {"trend": "unknown", "volatility": "medium", "favorable": False, "reason": "Données insuffisantes"}
            
            # 1. Analyser la tendance
            trend_analysis = self._analyze_trend(ohlcv)
            
            # 2. Analyser la volatilité
            volatility_analysis = self._analyze_volatility(ohlcv)
            
            # 3. Analyser le volume
            volume_analysis = self._analyze_volume(ohlcv)
            
            # 4. Analyser les niveaux clés
            levels_analysis = self._analyze_key_levels(ohlcv)
            
            # 5. Déterminer le régime de marché
            regime = self._determine_market_regime(trend_analysis, volatility_analysis)
            
            # 6. Évaluer si le marché est favorable pour le trading
            is_favorable, reason = self._evaluate_market_favorability(
                trend_analysis, volatility_analysis, volume_analysis, regime
            )
            
            # Consolidation des résultats
            market_state = {
                "symbol": symbol,
                "timeframe": timeframe,
                "timestamp": datetime.now(),
                "trend": trend_analysis["overall_trend"],
                "trend_strength": trend_analysis["strength"],
                "volatility": volatility_analysis["level"],
                "volume_state": volume_analysis["level"],
                "regime": regime,
                "key_levels": levels_analysis["key_levels"],
                "favorable": is_favorable,
                "reason": reason,
                "details": {
                    "trend": trend_analysis,
                    "volatility": volatility_analysis,
                    "volume": volume_analysis,
                    "levels": levels_analysis
                }
            }
            
            # Mettre à jour le cache
            self.market_states[cache_key] = (datetime.now(), market_state)
            
            # Mettre à jour le régime de marché
            self.market_regimes[symbol] = regime
            
            logger.info(f"Analyse du marché pour {symbol} ({timeframe}): Tendance={trend_analysis['overall_trend']}, "
                      f"Volatilité={volatility_analysis['level']}, Régime={regime}, Favorable={is_favorable}")
            
            return market_state
        
        except Exception as e:
            logger.error(f"Erreur lors de l'analyse du marché pour {symbol} ({timeframe}): {e}")
            return {"trend": "unknown", "volatility": "medium", "favorable": False, "reason": "Erreur d'analyse"}
    
    def _analyze_trend(self, ohlcv: pd.DataFrame) -> Dict:
        """
        Analyse la tendance du marché à partir des données OHLCV
        
        Args:
            ohlcv: DataFrame avec les données OHLCV
            
        Returns:
            Dictionnaire avec les résultats de l'analyse de tendance
        """
        trend = detect_trend(ohlcv)
        return trend
    
    def _analyze_volatility(self, ohlcv: pd.DataFrame) -> Dict:
        """
        Analyse la volatilité du marché à partir des données OHLCV
        
        Args:
            ohlcv: DataFrame avec les données OHLCV
            
        Returns:
            Dictionnaire avec les résultats de l'analyse de volatilité
        """
        volatility = calculate_volatility(ohlcv)
        return volatility
    
    def _analyze_volume(self, ohlcv: pd.DataFrame) -> Dict:
        """
        Analyse le volume du marché à partir des données OHLCV
        
        Args:
            ohlcv: DataFrame avec les données OHLCV
            
        Returns:
            Dictionnaire avec les résultats de l'analyse de volume
        """
        volume = ohlcv["volume"].rolling(window=20).mean().iloc[-1]
        avg_volume = ohlcv["volume"].mean()
        
        volume_state = "normal"
        if volume < avg_volume * self.thresholds["volume"]["low"]:
            volume_state = "low"
        elif volume > avg_volume * self.thresholds["volume"]["high"]:
            volume_state = "high"
        
        return {"level": volume_state, "current_volume": volume, "average_volume": avg_volume}
    
    def _analyze_key_levels(self, ohlcv: pd.DataFrame) -> Dict:
        """
        Analyse les niveaux clés du marché à partir des données OHLCV
        
        Args:
            ohlcv: DataFrame avec les données OHLCV
            
        Returns:
            Dictionnaire avec les résultats de l'analyse des niveaux clés
        """
        # Exemple simplifié d'analyse des niveaux clés
        support = ohlcv["low"].min()
        resistance = ohlcv["high"].max()
        
        return {"key_levels": {"support": support, "resistance": resistance}}
    
    def _determine_market_regime(self, trend_analysis: Dict, volatility_analysis: Dict) -> str:
        """
        Détermine le régime de marché à partir des analyses de tendance et de volatilité
        
        Args:
            trend_analysis: Résultats de l'analyse de tendance
            volatility_analysis: Résultats de l'analyse de volatilité
            
        Returns:
            Régime de marché (ex: "bullish", "bearish", "sideways")
        """
        if trend_analysis["overall_trend"] == "up" and volatility_analysis["level"] == "low":
            return "bullish"
        elif trend_analysis["overall_trend"] == "down" and volatility_analysis["level"] == "high":
            return "bearish"
        else:
            return "sideways"
    
    def _evaluate_market_favorability(self, trend_analysis: Dict, volatility_analysis: Dict, volume_analysis: Dict, regime: str) -> (bool, str):
        """
        Évalue si le marché est favorable pour le trading à partir des analyses de tendance, de volatilité et de volume
        
        Args:
            trend_analysis: Résultats de l'analyse de tendance
            volatility_analysis: Résultats de l'analyse de volatilité
            volume_analysis: Résultats de l'analyse de volume
            regime: Régime de marché
            
        Returns:
            Tuple (favorable: bool, reason: str)
        """
        if regime == "bullish" and volume_analysis["level"] != "low":
            return True, "Marché haussier avec volume adéquat"
        elif regime == "bearish" and volume_analysis["level"] != "low":
            return True, "Marché baissier avec volume adéquat"
        else:
            return False, "Marché non favorable"