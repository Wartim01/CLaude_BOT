# strategies/market_state.py
"""
Analyseur de l'état du marché
"""
import logging
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Union
from datetime import datetime, timedelta

def detect_divergence(ohlcv: pd.DataFrame, rsi: pd.Series, lookback: int = 10) -> Dict[str, bool]:
    """
    Détecte les divergences haussières entre le prix et le RSI
    
    Args:
        ohlcv: DataFrame avec les données OHLCV
        rsi: Series contenant les valeurs RSI
        lookback: Nombre de périodes à analyser
        
    Returns:
        Dict avec le résultat de l'analyse des divergences
    """
    # Prendre les n dernières périodes
    price_lows = ohlcv['low'].tail(lookback)
    rsi_values = rsi.tail(lookback)
    
    # Trouver les plus bas
    price_min = np.min(price_lows)
    price_min_idx = price_lows.idxmin()
    rsi_min = np.min(rsi_values)
    rsi_min_idx = rsi_values.idxmin()
    
    # Détecter divergence haussière (prix fait un plus bas mais pas le RSI)
    bullish = (price_min_idx > rsi_min_idx and 
              price_lows.iloc[-1] <= price_min * 1.02 and 
              rsi_values.iloc[-1] > rsi_min * 1.02)
    
    return {"bullish": bullish}

from config.config import PRIMARY_TIMEFRAME, SECONDARY_TIMEFRAMES
from config.trading_params import ADX_THRESHOLD, MARKET_COOLDOWN_PERIOD
from utils.logger import setup_logger

logger = setup_logger("market_state")

class MarketStateAnalyzer:
    """
    Analyse l'état du marché pour déterminer si les conditions sont favorables au trading
    """
    def __init__(self, data_fetcher):
        self.data_fetcher = data_fetcher
        self.unfavorable_since = {}  # {symbol: timestamp}
    
    def analyze_market_state(self, symbol: str) -> Dict:
        """
        Analyse l'état du marché pour un symbole donné
        
        Args:
            symbol: Paire de trading
            
        Returns:
            Dictionnaire avec l'analyse de l'état du marché
        """
        # Récupérer les données de marché
        market_data = self.data_fetcher.get_market_data(symbol)
        
        # Vérifier si des données sont disponibles
        if market_data["primary_timeframe"].get("ohlcv") is None or market_data["primary_timeframe"].get("ohlcv").empty:
            logger.warning(f"Données de marché non disponibles pour {symbol}")
            return {
                "favorable": False,
                "reason": "Données non disponibles",
                "cooldown": False,
                "details": {}
            }
        
        # Extraire les données et indicateurs
        ohlcv = market_data["primary_timeframe"]["ohlcv"]
        indicators = market_data["primary_timeframe"].get("indicators", {})
        
        # Analyse de l'état du marché
        market_state = {
            "favorable": True,  # Par défaut, considérer le marché comme favorable
            "reason": "Conditions normales",
            "cooldown": False,
            "details": {}
        }
        
        # 1. Vérifier la force de la tendance avec ADX
        adx_data = indicators.get("adx", {})
        if adx_data and "adx" in adx_data:
            adx_value = adx_data["adx"].iloc[-1]
            plus_di = adx_data["plus_di"].iloc[-1]
            minus_di = adx_data["minus_di"].iloc[-1]
            
            strong_trend = adx_value > ADX_THRESHOLD
            bearish_trend = minus_di > plus_di
            
            market_state["details"]["adx"] = {
                "value": float(adx_value),
                "plus_di": float(plus_di),
                "minus_di": float(minus_di),
                "strong_trend": bool(strong_trend),
                "bearish_trend": bool(bearish_trend)
            }
            
            # Si forte tendance baissière, marché défavorable
            if strong_trend and bearish_trend:
                market_state["favorable"] = False
                market_state["reason"] = "Forte tendance baissière"
        
        # 2. Vérifier l'alignement des EMA
        ema_data = indicators.get("ema", {})
        if ema_data:
            ema_short = ema_data.get("ema_9", pd.Series()).iloc[-1] if "ema_9" in ema_data else None
            ema_medium = ema_data.get("ema_21", pd.Series()).iloc[-1] if "ema_21" in ema_data else None
            ema_long = ema_data.get("ema_50", pd.Series()).iloc[-1] if "ema_50" in ema_data else None
            ema_baseline = ema_data.get("ema_200", pd.Series()).iloc[-1] if "ema_200" in ema_data else None
            
            current_price = ohlcv["close"].iloc[-1]
            
            # Vérifier l'alignement baissier
            if (ema_short is not None and ema_medium is not None and 
                ema_long is not None and ema_baseline is not None):
                
                bearish_alignment = (ema_short < ema_medium < ema_long < ema_baseline)
                price_below_baseline = current_price < ema_baseline
                
                market_state["details"]["ema_alignment"] = {
                    "bearish_alignment": bool(bearish_alignment),
                    "price_below_baseline": bool(price_below_baseline)
                }
                
                # Si prix sous les EMA importantes, marché défavorable
                if bearish_alignment and price_below_baseline:
                    # Vérifier si les conditions étaient déjà défavorables
                    if market_state["favorable"]:
                        market_state["favorable"] = False
                        market_state["reason"] = "Alignement baissier des EMA"
        
        # 3. Vérifier les bandes de Bollinger
        bb_data = indicators.get("bollinger", {})
        if bb_data:
            bb_middle = bb_data.get("middle", pd.Series()).iloc[-1] if "middle" in bb_data else None
            bb_bandwidth = bb_data.get("bandwidth", pd.Series()).iloc[-1] if "bandwidth" in bb_data else None
            
            # Vérifier si la volatilité est excessive
            if bb_bandwidth is not None:
                high_volatility = bb_bandwidth > 0.1  # Seuil arbitraire, à ajuster
                
                market_state["details"]["bollinger"] = {
                    "bandwidth": float(bb_bandwidth),
                    "high_volatility": bool(high_volatility)
                }
        
        # 4. Vérifier le RSI pour les conditions de survente/surachat
        rsi_data = indicators.get("rsi", None)
        if rsi_data is not None:
            rsi_value = rsi_data.iloc[-1]
            
            market_state["details"]["rsi"] = {
                "value": float(rsi_value)
            }
        
        # 5. Vérifier s'il faut mettre en place un cooldown
        if not market_state["favorable"]:
            current_time = datetime.now()
            
            # Si le marché vient de devenir défavorable, enregistrer le timestamp
            if symbol not in self.unfavorable_since:
                self.unfavorable_since[symbol] = current_time
                logger.info(f"Marché défavorable pour {symbol}, début du cooldown")
            
            # Vérifier si le cooldown est toujours actif
            cooldown_end = self.unfavorable_since[symbol] + timedelta(minutes=MARKET_COOLDOWN_PERIOD)
            if current_time < cooldown_end:
                market_state["cooldown"] = True
                minutes_remaining = int((cooldown_end - current_time).total_seconds() / 60)
                market_state["cooldown_remaining"] = minutes_remaining
                market_state["reason"] += f" (Cooldown: {minutes_remaining} min restantes)"
        else:
            # Si le marché est favorable, réinitialiser le cooldown
            if symbol in self.unfavorable_since:
                del self.unfavorable_since[symbol]
        
        # Ajouter un filtre de tendance
        ema_data = indicators.get("ema", {})
        if ema_data:
            price = ohlcv["close"].iloc[-1]
            ema_200 = ema_data.get("ema_200", pd.Series()).iloc[-1] if "ema_200" in ema_data else None
            
            # Ne trader à l'achat que si le prix est au-dessus de la EMA200
            if ema_200 is not None and price < ema_200:
                market_state["favorable"] = False
                market_state["reason"] = "Prix sous la EMA200, éviter les longs"
        
        # Ajouter un filtre de volatilité
        if "bollinger" in indicators:
            bandwidth = indicators["bollinger"].get("bandwidth", pd.Series()).iloc[-1]
            if bandwidth > 0.06:  # Seuil de volatilité élevée
                market_state["favorable"] = False
                market_state["reason"] = "Volatilité trop élevée"
        
        return market_state
    
    def check_for_reversal(self, symbol: str) -> Dict:
        """
        Recherche des signaux de retournement pour sortir d'un cooldown
        
        Args:
            symbol: Paire de trading
            
        Returns:
            Dictionnaire avec les signaux de retournement
        """
        # Récupérer les données de marché
        market_data = self.data_fetcher.get_market_data(symbol)
        
        # Vérifier si des données sont disponibles
        if market_data["primary_timeframe"].get("ohlcv") is None or market_data["primary_timeframe"].get("ohlcv").empty:
            return {
                "reversal_detected": False,
                "confidence": 0,
                "reason": "Données non disponibles"
            }
        
        # Extraire les données et indicateurs
        ohlcv = market_data["primary_timeframe"]["ohlcv"]
        indicators = market_data["primary_timeframe"].get("indicators", {})
        
        # Initialiser les signaux de retournement
        reversal_signals = []
        confidence = 0
        
        # 1. Vérifier le RSI pour les conditions de survente
        rsi_data = indicators.get("rsi", None)
        if rsi_data is not None:
            rsi_value = rsi_data.iloc[-1]
            rsi_prev = rsi_data.iloc[-2] if len(rsi_data) > 1 else None
            
            # RSI en zone de survente
            if rsi_value < 30:
                reversal_signals.append("RSI en zone de survente")
                confidence += 20
            
            # RSI qui remonte depuis la zone de survente
            if rsi_prev is not None and rsi_prev < 30 and rsi_value > rsi_prev:
                reversal_signals.append("RSI remonte depuis la zone de survente")
                confidence += 15
        
        # 2. Vérifier les bougies de retournement
        if len(ohlcv) >= 3:
            current_candle = {
                "open": ohlcv["open"].iloc[-1],
                "high": ohlcv["high"].iloc[-1],
                "low": ohlcv["low"].iloc[-1],
                "close": ohlcv["close"].iloc[-1]
            }
            
            prev_candle = {
                "open": ohlcv["open"].iloc[-2],
                "high": ohlcv["high"].iloc[-2],
                "low": ohlcv["low"].iloc[-2],
                "close": ohlcv["close"].iloc[-2]
            }
            
            prev_prev_candle = {
                "open": ohlcv["open"].iloc[-3],
                "high": ohlcv["high"].iloc[-3],
                "low": ohlcv["low"].iloc[-3],
                "close": ohlcv["close"].iloc[-3]
            }
            
            # Vérifier le marteau ou étoile du matin
            if (prev_candle["close"] < prev_candle["open"] and  # Bougie baissière
                current_candle["close"] > current_candle["open"] and  # Bougie haussière
                current_candle["close"] > prev_candle["open"]):  # Clôture au-dessus de l'ouverture précédente
                
                reversal_signals.append("Motif de retournement haussier")
                confidence += 25
            
            # Vérifier le double bottom
            if (prev_prev_candle["low"] < prev_prev_candle["open"] and
                prev_candle["low"] <= prev_prev_candle["low"] * 1.01 and  # Deuxième creux similaire
                current_candle["close"] > prev_candle["high"]):  # Cassure haussière
                
                reversal_signals.append("Double bottom potentiel")
                confidence += 30
        
        # 3. Vérifier la divergence haussière sur le RSI
        if rsi_data is not None and len(ohlcv) >= 10:
            divergence = detect_divergence(ohlcv, rsi_data)
            
            if divergence["bullish"]:
                reversal_signals.append("Divergence haussière RSI")
                confidence += 35
        
        # 4. Vérifier le croisement des EMA courtes
        ema_data = indicators.get("ema", {})
        if "ema_9" in ema_data and "ema_21" in ema_data:
            ema_short = ema_data["ema_9"]
            ema_medium = ema_data["ema_21"]
            
            if len(ema_short) >= 2 and len(ema_medium) >= 2:
                current_cross = ema_short.iloc[-1] > ema_medium.iloc[-1]
                prev_cross = ema_short.iloc[-2] <= ema_medium.iloc[-2]
                
                if current_cross and prev_cross:
                    reversal_signals.append("Croisement EMA 9/21 haussier")
                    confidence += 20
        
        # Résultat final
        reversal_detected = confidence >= 50  # Seuil de confiance
        
        return {
            "reversal_detected": reversal_detected,
            "confidence": confidence,
            "signals": reversal_signals,
            "details": {
                "rsi": float(rsi_data.iloc[-1]) if rsi_data is not None else None
            }
        }