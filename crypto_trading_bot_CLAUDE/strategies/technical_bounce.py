
# strategies/technical_bounce.py
"""
Stratégie de rebond technique
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Union
from datetime import datetime

from strategies.strategy_base import StrategyBase
from config.trading_params import (
    RSI_OVERSOLD,
    STOP_LOSS_PERCENT,
    TAKE_PROFIT_PERCENT
)
from utils.logger import setup_logger

# Importer la fonction de détection de divergence
from indicators.momentum import detect_divergence

logger = setup_logger("technical_bounce")

class TechnicalBounceStrategy(StrategyBase):
    """
    Stratégie qui cherche à capturer les rebonds techniques après des baisses de prix
    """
    def __init__(self, data_fetcher, market_analyzer, scoring_engine):
        super().__init__(data_fetcher, market_analyzer, scoring_engine)
    
    def find_trading_opportunity(self, symbol: str) -> Optional[Dict]:
        """
        Cherche une opportunité de rebond technique pour le symbole donné
        
        Args:
            symbol: Paire de trading
            
        Returns:
            Opportunité de trading ou None si aucune opportunité n'est trouvée
        """
        # Récupérer les données de marché
        market_data = self.data_fetcher.get_market_data(symbol)
        
        # Vérifier si des données sont disponibles
        if (market_data["primary_timeframe"].get("ohlcv") is None or 
            market_data["primary_timeframe"].get("ohlcv").empty):
            logger.warning(f"Données de marché non disponibles pour {symbol}")
            return None
        
        # Extraire les données et indicateurs
        ohlcv = market_data["primary_timeframe"]["ohlcv"]
        indicators = market_data["primary_timeframe"].get("indicators", {})
        
        # Vérifier l'état du marché
        market_state = self.market_analyzer.analyze_market_state(symbol)
        if not market_state["favorable"] or market_state["cooldown"]:
            return None
        
        # Rechercher des signaux de rebond technique
        bounce_signals = self._detect_bounce_signals(symbol, ohlcv, indicators)
        
        # Si aucun signal de rebond n'est trouvé, retourner None
        if not bounce_signals["bounce_detected"]:
            return None
        
        # Calculer les niveaux d'entrée, de stop-loss et de take-profit
        current_price = ohlcv["close"].iloc[-1]
        
        # Pour un ordre long (achat)
        entry_price = current_price
        stop_loss = entry_price * (1 - STOP_LOSS_PERCENT/100)
        take_profit = entry_price * (1 + TAKE_PROFIT_PERCENT/100)
        
        # Calculer le score de l'opportunité
        opportunity_score = self._calculate_opportunity_score(bounce_signals, market_state, ohlcv, indicators)
        
        # Générer une explication textuelle
        reasoning = self._generate_reasoning(bounce_signals, market_state, opportunity_score)
        
        # Créer l'opportunité de trading
        opportunity = {
            "symbol": symbol,
            "strategy": "technical_bounce",
            "side": "BUY",  # Cette stratégie ne prend que des positions longues
            "entry_price": entry_price,
            "stop_loss": stop_loss,
            "take_profit": take_profit,
            "score": opportunity_score,
            "reasoning": reasoning,
            "signals": bounce_signals,
            "market_conditions": market_state,
            "timestamp": datetime.now(),
            "indicators": {
                "rsi": float(indicators["rsi"].iloc[-1]) if "rsi" in indicators else None,
                "bollinger": {
                    "lower": float(indicators["bollinger"]["lower"].iloc[-1]) if "bollinger" in indicators else None,
                    "percent_b": float(indicators["bollinger"]["percent_b"].iloc[-1]) if "bollinger" in indicators else None
                }
            }
        }
        
        logger.info(f"Opportunité de rebond technique trouvée pour {symbol} (score: {opportunity_score})")
        return opportunity
    
    def _detect_bounce_signals(self, symbol: str, ohlcv: pd.DataFrame, indicators: Dict) -> Dict:
        """
        Détecte les signaux de rebond technique avec critères améliorés
        
        Args:
            symbol: Paire de trading
            ohlcv: DataFrame avec les données OHLCV
            indicators: Dictionnaire des indicateurs techniques
            
        Returns:
            Dictionnaire avec les signaux de rebond détectés
        """
        bounce_signals = {
            "bounce_detected": False,
            "signals": [],
            "strength": 0,
            "volume_ratio": 1.0,  # Par défaut
            "multi_timeframe_confirmation": 0  # Nouveau: nombre de timeframes confirmant
        }
        
        # Vérifier la présence des indicateurs nécessaires
        if "rsi" not in indicators or "bollinger" not in indicators:
            return bounce_signals
          # Détection de la tendance de marché
        trend_direction = self._detect_market_trend(ohlcv, indicators)
        
        # Seuil de force pour détecter un rebond - RÉDUIT pour être moins strict
        strength_threshold = 35  # Était probablement plus élevé, réduit à 35
        # Extraire les indicateurs
        rsi = indicators["rsi"]
        bollinger = indicators["bollinger"]
        
        # NOUVEAU: Analyser le contexte de marché et filtrer les conditions défavorables
        trend_direction = self._detect_market_trend(ohlcv, indicators)
        if trend_direction == "strong_bearish":
            # Si tendance fortement baissière, exiger des signaux plus forts
            bounce_signals["trend_context"] = "strong_bearish"
            strength_threshold = 70  # Exiger des signaux plus forts dans un marché baissier
        else:
            strength_threshold = 40  # Seuil normal
        
        # NOUVEAU: Analyse de la structure de prix
        price_structure = self._analyze_price_structure(ohlcv)
        if price_structure.get("double_bottom", False):
            bounce_signals["signals"].append("Structure de double fond détectée")
            bounce_signals["strength"] += 25
        
        # NOUVEAU: Vérification des niveaux de support
        support_test = self._check_support_test(ohlcv, indicators)
        if support_test["support_tested"]:
            bounce_signals["signals"].append(f"Test de support à {support_test['support_level']:.2f}")
            bounce_signals["strength"] += 20
        
        # 1. Vérifier le RSI en zone de survente
        if len(rsi) >= 2:
            rsi_current = rsi.iloc[-1]
            rsi_prev = rsi.iloc[-2]
            
            # ASSOUPLI: Seuil de RSI légèrement augmenté pour capturer plus de signaux
            oversold_condition = rsi_current < 32  # Légèrement plus permissif (standard est 30)
            rsi_turning_up = rsi_current > rsi_prev and rsi_prev < 35  # Moins strict sur le seuil
            
            if oversold_condition:
                bounce_signals["signals"].append("RSI en zone de survente")
                bounce_signals["strength"] += 20  # Augmenté pour donner plus d'importance
            
            if rsi_turning_up:
                bounce_signals["signals"].append("RSI remonte depuis zone basse")
                bounce_signals["strength"] += 15
        
        # 2. Vérifier les bandes de Bollinger
        if "percent_b" in bollinger and len(bollinger["percent_b"]) >= 2:
            percent_b_current = bollinger["percent_b"].iloc[-1]
            percent_b_prev = bollinger["percent_b"].iloc[-2]
            
            # ASSOUPLI: Seuils légèrement ajustés
            price_below_lower_band = percent_b_current < 0.05  # Était 0
            price_returning_to_band = percent_b_current > percent_b_prev and percent_b_prev < 0.1  # Était probablement plus strict
            
            if price_below_lower_band:
                bounce_signals["signals"].append("Prix sous la bande inférieure de Bollinger")
                bounce_signals["strength"] += 20
            
            if price_returning_to_band:
                bounce_signals["signals"].append("Prix remonte vers la bande inférieure")
                bounce_signals["strength"] += 15
        
        # 3. Vérifier les mèches (wicks) des chandeliers
        if len(ohlcv) >= 2:
            current_candle = ohlcv.iloc[-1]
            prev_candle = ohlcv.iloc[-2]
            
            current_body = abs(current_candle["close"] - current_candle["open"])
            current_total_range = current_candle["high"] - current_candle["low"]
            current_lower_wick = min(current_candle["open"], current_candle["close"]) - current_candle["low"]
            
            # ASSOUPLI: Seuil légèrement réduit pour les mèches inférieures
            if current_total_range > 0 and current_lower_wick / current_total_range > 0.4:  # Était 0.5
                bounce_signals["signals"].append("Mèche inférieure significative (rejet)")
                bounce_signals["strength"] += 18
            
            # Vérifier si le chandelier actuel est haussier après un chandelier baissier
            current_bullish = current_candle["close"] > current_candle["open"]
            prev_bearish = prev_candle["close"] < prev_candle["open"]
            
            if current_bullish and prev_bearish:
                bounce_signals["signals"].append("Chandelier haussier après chandelier baissier")
                bounce_signals["strength"] += 15
        
        # 4. Vérifier les divergences haussières - conservé tel quel
        if "rsi" in indicators and len(ohlcv) >= 10:
            from indicators.momentum import detect_divergence
            divergence = detect_divergence(ohlcv, rsi)
            
            if divergence["bullish"]:
                bounce_signals["signals"].append("Divergence haussière RSI détectée")
                bounce_signals["strength"] += 25  # Augmenté car signal fiable
        
        # 5. Vérifier les pics de volume
        if len(ohlcv) >= 5:
            # Calculer la moyenne des volumes récents (sauf le dernier)
            avg_volume = ohlcv['volume'].iloc[-5:-1].mean()
            current_volume = ohlcv['volume'].iloc[-1]
            volume_ratio = current_volume / avg_volume if avg_volume > 0 else 1.0
            
            # ASSOUPLI: Seuil réduit pour les volumes
            if volume_ratio > 1.7:  # Était probablement 2.0
                # Vérifier si c'est un volume de capitulation avec clôture haussière
                if ohlcv['close'].iloc[-1] > ohlcv['open'].iloc[-1]:
                    bounce_signals["signals"].append("Volume élevé avec clôture haussière")
                    bounce_signals["strength"] += 18
                    bounce_signals["volume_ratio"] = volume_ratio
        
        # Déterminer si un rebond est détecté
        # ASSOUPLI: Exigences réduites pour la détection de rebond
        bounce_signals["bounce_detected"] = (
            len(bounce_signals["signals"]) >= 2 and  # Maintenu à 2 minimum
            bounce_signals["strength"] >= strength_threshold  # Seuil réduit
        )
        
        # MODIFICATION IMPORTANTE: Moins strict sur les conditions de marché en tendance baissière
        if trend_direction == "bearish" or trend_direction == "strong_bearish":
            # Ne pas rejeter automatiquement en tendance baissière
            # mais exiger plus de signaux de confirmation
            if bounce_signals["bounce_detected"] and len(bounce_signals["signals"]) < 3:
                bounce_signals["bounce_detected"] = False        
        # 6. Vérifier les patterns de reversal
        if len(ohlcv) >= 3:
            # Vérifier le pattern de hammer (marteau)
            last_candle = ohlcv.iloc[-1]
            body_size = abs(last_candle['close'] - last_candle['open'])
            total_range = last_candle['high'] - last_candle['low']
            lower_wick = min(last_candle['open'], last_candle['close']) - last_candle['low']
            
            # Un marteau a une petite tête et une longue mèche basse
            if total_range > 0 and body_size / total_range < 0.3 and lower_wick / total_range > 0.6:
                bounce_signals["signals"].append("Pattern de marteau détecté")
                bounce_signals["strength"] += 20
        
        # Déterminer si un rebond est détecté
        bounce_signals["bounce_detected"] = len(bounce_signals["signals"]) >= 2 and bounce_signals["strength"] >= 40
        
        # Vérifier la convergence de plusieurs indicateurs
        signal_count = len(bounce_signals["signals"])
        
        # Renforcer les critères pour détecter un rebond
        if signal_count >= 3:  # Exiger au moins 3 signaux au lieu de 2
            bounce_signals["bounce_detected"] = True
        else:
            bounce_signals["bounce_detected"] = False
        
        # Vérifier la confirmation sur plusieurs timeframes
        if bounce_signals["bounce_detected"]:
            # Obtenir des données du timeframe supérieur
            higher_tf_data = self.data_fetcher.get_market_data(symbol)["secondary_timeframes"]
            
            if bounce_signals["bounce_detected"]:
                # Obtenir des données du timeframe supérieur
                market_data = self.data_fetcher.get_market_data(symbol)
                higher_tf_data = market_data.get("secondary_timeframes", {})
                
                # Vérifier si le timeframe supérieur confirme aussi un rebond
                if not self._check_higher_timeframe_confirmation(higher_tf_data):
                    bounce_signals["bounce_detected"] = False
                    bounce_signals["signals"].append("Non confirmé sur timeframe supérieur")
         # NOUVEAU: Vérifier la confirmation sur plusieurs timeframes
        tf_confirmations = self._check_higher_timeframe_confirmation(symbol)
        bounce_signals["multi_timeframe_confirmation"] = tf_confirmations
        
        # AMÉLIORÉ: Analyse de volume plus sophistiquée
        volume_analysis = self._analyze_volume_pattern(ohlcv)
        if volume_analysis["volume_spike"]:
            bounce_signals["signals"].append("Pic de volume haussier")
            bounce_signals["strength"] += 15
            bounce_signals["volume_ratio"] = volume_analysis["volume_ratio"]
            
            # Volume climax après une forte baisse (capitulation)
            if volume_analysis["capitulation"]:
                bounce_signals["signals"].append("Volume de capitulation détecté")
                bounce_signals["strength"] += 15
        
        # NOUVEAU: Calculer le score de confiance
        confidence_score = bounce_signals["strength"]
        
        # Ajuster en fonction des confirmations multi-timeframe
        confidence_score += tf_confirmations * 5
        
        # Pénaliser en cas de tendance fortement baissière sans volume de capitulation
        if trend_direction == "strong_bearish" and not volume_analysis.get("capitulation", False):
            confidence_score *= 0.7
        
        # Déterminer si un rebond est détecté avec des critères plus stricts
        bounce_signals["bounce_detected"] = (
            len(bounce_signals["signals"]) >= 3 and confidence_score >= strength_threshold
        )
        return bounce_signals
    
    def _check_higher_timeframe_confirmation(self, higher_tf_data: Dict) -> bool:
        """
        Vérifie si les timeframes supérieurs confirment également un signal de rebond
        
        Args:
            higher_tf_data: Données des timeframes supérieurs
            
        Returns:
            True si confirmé, False sinon
        """
        # Par défaut, considérer comme confirmé si aucune donnée n'est disponible
        if not higher_tf_data:
            return True
        
        confirmation_count = 0
        timeframes_checked = 0
        
        # Parcourir les timeframes supérieurs (1h, 4h)
        for tf, tf_data in higher_tf_data.items():
            if "ohlcv" not in tf_data or tf_data["ohlcv"].empty:
                continue
            
            timeframes_checked += 1
            ohlcv = tf_data["ohlcv"]
            indicators = tf_data.get("indicators", {})
            
            # Vérifier le RSI
            if "rsi" in indicators:
                rsi = indicators["rsi"]
                if len(rsi) >= 2:
                    rsi_current = rsi.iloc[-1]
                    rsi_prev = rsi.iloc[-2]
                    
                    if rsi_current > rsi_prev and rsi_current < 50:
                        # RSI en hausse mais encore sous 50 = bon signe pour un rebond
                        confirmation_count += 1
            
            # Vérifier les bandes de Bollinger
            if "bollinger" in indicators and "percent_b" in indicators["bollinger"]:
                percent_b = indicators["bollinger"]["percent_b"]
                if len(percent_b) >= 2:
                    percent_b_current = percent_b.iloc[-1]
                    percent_b_prev = percent_b.iloc[-2]
                    
                    if percent_b_current > percent_b_prev and percent_b_current < 0.5:
                        # %B en hausse mais encore sous 0.5 = bon signe pour un rebond
                        confirmation_count += 1
            
            # Vérifier le pattern de chandelier
            if len(ohlcv) >= 3:
                last_candle = ohlcv.iloc[-1]
                prev_candle = ohlcv.iloc[-2]
                
                if last_candle["close"] > last_candle["open"] and prev_candle["close"] < prev_candle["open"]:
                    # Bougie haussière après bougie baissière = bon signe pour un rebond
                    confirmation_count += 1
        
        # Considérer comme confirmé si au moins la moitié des vérifications sont positives
        # et qu'au moins un timeframe a été vérifié
        return timeframes_checked > 0 and confirmation_count >= timeframes_checked / 2     

    def _calculate_opportunity_score(self, bounce_signals: Dict, market_state: Dict,
                                ohlcv: pd.DataFrame, indicators: Dict) -> int:
        """
        Calcule le score de l'opportunité de trading
        
        Args:
            bounce_signals: Signaux de rebond détectés
            market_state: État du marché
            ohlcv: DataFrame avec les données OHLCV
            indicators: Dictionnaire des indicateurs techniques
            
        Returns:
            Score de l'opportunité (0-100)
        """
        # Utiliser le moteur de scoring pour calculer le score
        score_data = {
            "bounce_signals": bounce_signals,
            "market_state": market_state,
            "ohlcv": ohlcv,
            "indicators": indicators
        }
        
        # Appeler le scoring engine et vérifier le résultat
        score_result = self.scoring_engine.calculate_score(score_data, "technical_bounce")
        
        # Vérifier si score_result est None ou ne contient pas la clé 'score'
        if score_result is None:
            # Logging pour diagnostic
            logger.error("Le scoring engine a retourné None au lieu d'un résultat valide")
            return 0  # Score par défaut si erreur
        
        # Vérifier si la clé 'score' existe dans score_result
        if "score" not in score_result:
            logger.error(f"Résultat du scoring incomplet: {score_result}")
            return 0  # Score par défaut si erreur
        
        return score_result["score"]
    
    
    def _generate_reasoning(self, bounce_signals: Dict, market_state: Dict, score: int) -> str:
        """
        Génère une explication textuelle pour l'opportunité de trading
        
        Args:
            bounce_signals: Signaux de rebond détectés
            market_state: État du marché
            score: Score de l'opportunité
            
        Returns:
            Explication textuelle
        """
        signals_text = ", ".join(bounce_signals["signals"])
        
        reasoning = f"Opportunité de rebond technique détectée (score: {score}/100). "
        reasoning += f"Signaux: {signals_text}. "
        
        # Ajouter des détails sur l'état du marché
        if "details" in market_state:
            market_details = market_state["details"]
            
            if "rsi" in market_details:
                reasoning += f"RSI actuel: {market_details['rsi'].get('value', 'N/A'):.1f}. "
            
            if "bollinger" in market_details:
                reasoning += f"Volatilité: {market_details['bollinger'].get('bandwidth', 'N/A'):.3f}. "
            
            if "adx" in market_details:
                reasoning += f"Force de tendance (ADX): {market_details['adx'].get('value', 'N/A'):.1f}. "
        
        return reasoning
    def _get_rsi_oversold_duration(self, rsi: pd.Series) -> int:
        """
        Calcule la durée pendant laquelle le RSI est resté en zone de survente
        
        Returns:
            Nombre de périodes consécutives en zone de survente
        """
        duration = 0
        for i in range(len(rsi)-1, -1, -1):
            if rsi.iloc[i] < RSI_OVERSOLD:
                duration += 1
            else:
                break
        return duration

    def _analyze_volume_pattern(self, ohlcv: pd.DataFrame) -> Dict:
        """
        Analyse avancée des patterns de volume
        """
        if len(ohlcv) < 10:
            return {"volume_spike": False, "volume_ratio": 1.0}
        
        # Calculer la moyenne des volumes récents (sauf le dernier)
        avg_volume = ohlcv['volume'].iloc[-10:-1].mean()
        current_volume = ohlcv['volume'].iloc[-1]
        volume_ratio = current_volume / avg_volume if avg_volume > 0 else 1.0
        
        # Vérifier si le volume actuel est significativement plus élevé
        is_spike = volume_ratio > 2.0
        
        # Vérifier si c'est un volume de capitulation
        is_capitulation = False
        if is_spike and volume_ratio > 3.0:
            # Vérifier si le prix a chuté significativement avant ce volume
            price_drop = (ohlcv['close'].iloc[-3] - ohlcv['low'].iloc[-1]) / ohlcv['close'].iloc[-3]
            if price_drop > 0.05:  # Chute d'au moins 5%
                # Et si la clôture est plus haute que l'ouverture (rebond)
                if ohlcv['close'].iloc[-1] > ohlcv['open'].iloc[-1]:
                    is_capitulation = True
        
        return {
            "volume_spike": is_spike,
            "volume_ratio": volume_ratio,
            "capitulation": is_capitulation
        }

    def _analyze_price_structure(self, ohlcv: pd.DataFrame) -> Dict:
        """
        Analyse la structure de prix pour détecter les patterns de retournement
        """
        if len(ohlcv) < 20:
            return {"double_bottom": False}
        
        # Détecter double bottom (W-pattern)
        first_low = None
        second_low = None
        
        # Rechercher les deux derniers creux significatifs
        for i in range(len(ohlcv)-15, len(ohlcv)-1):
            if (ohlcv['low'].iloc[i] < ohlcv['low'].iloc[i-1] and 
                ohlcv['low'].iloc[i] < ohlcv['low'].iloc[i+1]):
                if first_low is None:
                    first_low = (i, ohlcv['low'].iloc[i])
                else:
                    second_low = (i, ohlcv['low'].iloc[i])
                    break
        
        if first_low and second_low:
            # Vérifier si les deux creux sont à des niveaux similaires (tolérance de 1%)
            price_diff = abs(first_low[1] - second_low[1]) / first_low[1]
            time_diff = second_low[0] - first_low[0]
            
            is_double_bottom = (price_diff < 0.01 and time_diff >= 5 and time_diff <= 20)
            
            return {"double_bottom": is_double_bottom}
        
        return {"double_bottom": False}

    def _check_support_test(self, ohlcv: pd.DataFrame, indicators: Dict) -> Dict:
        """
        Vérifie si le prix teste un niveau de support important
        """
        if len(ohlcv) < 50:
            return {"support_tested": False}
        
        # Identifier les niveaux de support potentiels
        support_levels = []
        
        # 1. EMA 200 comme support dynamique
        if "ema" in indicators and "ema_200" in indicators["ema"]:
            ema200 = indicators["ema"]["ema_200"].iloc[-1]
            current_price = ohlcv['close'].iloc[-1]
            
            # Vérifier si le prix est proche de l'EMA 200
            if abs(current_price - ema200) / ema200 < 0.01:
                return {
                    "support_tested": True,
                    "support_level": ema200,
                    "support_type": "EMA 200"
                }
        
        # 2. Détecter les niveaux de support statiques basés sur les creux précédents
        lows = []
        for i in range(20, len(ohlcv)-1):
            if (ohlcv['low'].iloc[i] < ohlcv['low'].iloc[i-1] and 
                ohlcv['low'].iloc[i] < ohlcv['low'].iloc[i+1]):
                lows.append(ohlcv['low'].iloc[i])
        
        # Regrouper les niveaux proches
        if lows:
            current_price = ohlcv['close'].iloc[-1]
            for low in lows:
                if abs(current_price - low) / low < 0.02:
                    return {
                        "support_tested": True,
                        "support_level": low,
                        "support_type": "Support historique"
                    }
        
        return {"support_tested": False}
    def _detect_market_trend(self, ohlcv: pd.DataFrame, indicators: Dict) -> str:
        """
        Détecte la direction et la force de la tendance actuelle du marché
        
        Args:
            ohlcv: DataFrame avec les données OHLCV
            indicators: Dictionnaire des indicateurs techniques
            
        Returns:
            Direction de la tendance ('strong_bullish', 'bullish', 'neutral', 'bearish', 'strong_bearish')
        """
        trend_scores = []  # Initialize the trend_scores list
        
        if len(ohlcv) < 20:
                return "neutral"  # Données insuffisantes
            
            # Méthode 1: Utiliser les EMA pour déterminer la tendance
        ema_data = indicators.get("ema", {})
        if "ema_9" in ema_data and "ema_21" in ema_data and "ema_50" in ema_data:
                ema_short = ema_data["ema_9"].iloc[-1]
                ema_medium = ema_data["ema_21"].iloc[-1]
                ema_long = ema_data["ema_50"].iloc[-1]
                
                # Vérifier l'alignement des EMA - version moins stricte
                if ema_short > ema_medium > ema_long:
                    # Tendance haussière
                    return "bullish"  # Moins strict: "bullish" au lieu de "strong_bullish"
                elif ema_short < ema_medium < ema_long:
                    # Tendance baissière
                    current_price = ohlcv['close'].iloc[-1]
                    # Moins strict: vérifier uniquement si le prix est sous l'EMA courte
                    if current_price < ema_short:
                        return "bearish"  # Moins strict: "bearish" au lieu de "strong_bearish"
            
            # Méthode 2: Utiliser l'ADX pour déterminer la force de la tendance - version moins stricte
        adx_data = indicators.get("adx", {})
        if "adx" in adx_data and "plus_di" in adx_data and "minus_di" in adx_data:
                adx = adx_data["adx"].iloc[-1]
                plus_di = adx_data["plus_di"].iloc[-1]
                minus_di = adx_data["minus_di"].iloc[-1]
                
                # Seuil ADX réduit pour être moins strict
                if adx > 20:  # Était 25, réduit à 20
                    if plus_di > minus_di:
                        return "bullish"  # Moins strict: "bullish" au lieu de "strong_bullish"
                    else:
                        return "bearish"  # Moins strict: "bearish" au lieu de "strong_bearish"
            
            # Méthode 3: Analyser la pente des prix récents - version moins stricte
        recent_closes = ohlcv['close'].tail(10).values
        if len(recent_closes) >= 10:
                # Calculer la pente linéaire
                x = np.arange(len(recent_closes))
                slope, _, _, _, _ = np.polyfit(x, recent_closes, 1, full=True)
                
                # Normaliser la pente par rapport au prix moyen
                avg_price = np.mean(recent_closes)
                norm_slope = slope[0] / avg_price * 100
                
                # Seuils réduits pour être moins stricts
                if norm_slope > 0.3:  # Était 0.5, réduit à 0.3
                    return "bullish"
                elif norm_slope < -0.3:  # Était -0.5, réduit à -0.3
                    return "bearish"
            
        
            # 4. NOUVEAU: Analyse des structures de prix
        price_structure_trend = self._get_price_structure_trend(ohlcv)
        trend_scores.append(price_structure_trend)
        
        # 5. NOUVEAU: Analyse des volumes
        volume_trend = self._get_volume_trend(ohlcv)
        trend_scores.append(volume_trend)
        
        # Calcul du score global pondéré
        # Strong bearish: -2, Bearish: -1, Neutral: 0, Bullish: 1, Strong bullish: 2
        weights = {
            "strong_bearish": -2,
            "bearish": -1,
            "neutral": 0,
            "bullish": 1,
            "strong_bullish": 2
        }
        
        # Pondération des méthodes (certaines sont plus fiables que d'autres)
        method_weights = {
            "ema": 0.3,        # EMA a un poids important
            "adx": 0.25,       # ADX également
            "price_slope": 0.15,
            "price_structure": 0.15,
            "volume": 0.15
        }
        
        # Calcul du score final pondéré
        weighted_score = 0
        for i, trend in enumerate(trend_scores):
            method_name = ["ema", "adx", "price_slope", "price_structure", "volume"][i]
            weighted_score += weights.get(trend, 0) * method_weights.get(method_name, 0.1)
        
        # Détermination finale de la tendance basée sur le score
        if weighted_score >= 1.2:
            return "strong_bullish"
        elif weighted_score >= 0.5:
            return "bullish"
        elif weighted_score <= -1.2:
            return "strong_bearish"
        elif weighted_score <= -0.5:
            return "bearish"
        else:
            return "neutral"

    
    def _check_higher_timeframe_confirmation(self, symbol: str) -> int:
        """
        Vérifie le nombre de timeframes supérieurs qui confirment le signal de rebond
        
        Args:
            symbol: Paire de trading
            
        Returns:
            Nombre de timeframes confirmant le signal (0-2)
        """
        # Dans un contexte de backtest, simuler une confirmation
        return 1  # Par défaut, considérer qu'un timeframe confirme

    def _analyze_price_structure(self, ohlcv: pd.DataFrame) -> Dict:
        """
        Analyse la structure de prix pour détecter les patterns de retournement
        
        Args:
            ohlcv: DataFrame avec les données OHLCV
            
        Returns:
            Dictionnaire avec les patterns détectés
        """
        if len(ohlcv) < 20:
            return {"double_bottom": False}
        
        # Recherche simplifiée de double bottom
        is_double_bottom = False
        
        # Vérifier si les 5 dernières bougies montrent une reprise après un creux
        if ohlcv['low'].iloc[-5:-3].min() < ohlcv['low'].iloc[-10:-5].min() and ohlcv['close'].iloc[-1] > ohlcv['close'].iloc[-5]:
            is_double_bottom = True
        
        return {"double_bottom": is_double_bottom}

    def _check_support_test(self, ohlcv: pd.DataFrame, indicators: Dict) -> Dict:
        """
        Vérifie si le prix teste un niveau de support important
        
        Args:
            ohlcv: DataFrame avec les données OHLCV
            indicators: Dictionnaire des indicateurs techniques
            
        Returns:
            Dictionnaire avec les informations de test de support
        """
        if len(ohlcv) < 20:
            return {"support_tested": False}
        
        # Version simplifiée: vérifier si le prix est proche d'un creux récent
        recent_low = ohlcv['low'].iloc[-20:].min()
        current_close = ohlcv['close'].iloc[-1]
        
        # Si le prix actuel est à moins de 2% du creux récent
        if abs(current_close - recent_low) / recent_low < 0.02:
            return {
                "support_tested": True,
                "support_level": recent_low,
                "support_type": "Support récent"
            }
        
        return {"support_tested": False}

    def _analyze_volume_pattern(self, ohlcv: pd.DataFrame) -> Dict:
        """
        Analyse les patterns de volume
        
        Args:
            ohlcv: DataFrame avec les données OHLCV
            
        Returns:
            Dictionnaire avec les analyses de volume
        """
        if len(ohlcv) < 10:
            return {"volume_spike": False, "volume_ratio": 1.0, "capitulation": False}
        
        # Calculer la moyenne des volumes récents
        avg_volume = ohlcv['volume'].iloc[-10:-1].mean()
        current_volume = ohlcv['volume'].iloc[-1]
        
        # Calculer le ratio de volume
        volume_ratio = current_volume / avg_volume if avg_volume > 0 else 1.0
        
        # Vérifier s'il y a un pic de volume
        is_spike = volume_ratio > 2.0
        
        # Vérifier s'il y a une capitulation (forte baisse suivie d'un fort volume et d'un rebond)
        is_capitulation = False
        if is_spike and volume_ratio > 3.0:
            price_drop = (ohlcv['high'].iloc[-3] - ohlcv['low'].iloc[-1]) / ohlcv['high'].iloc[-3]
            if price_drop > 0.03 and ohlcv['close'].iloc[-1] > ohlcv['open'].iloc[-1]:
                is_capitulation = True
        
        return {
            "volume_spike": is_spike,
            "volume_ratio": volume_ratio,
            "capitulation": is_capitulation
        }

    def _get_rsi_oversold_duration(self, rsi: pd.Series) -> int:
        """
        Calcule la durée pendant laquelle le RSI est resté en zone de survente
        
        Args:
            rsi: Série pandas contenant les valeurs du RSI
            
        Returns:
            Nombre de périodes en zone de survente
        """
        if rsi.empty:
            return 0
        
        # Compter le nombre de périodes où le RSI était en zone de survente
        oversold_count = 0
        
        # Parcourir les valeurs du RSI en partant de la dernière
        for i in range(len(rsi)-1, max(0, len(rsi)-10), -1):
            if rsi.iloc[i] < 30:  # RSI_OVERSOLD
                oversold_count += 1
            else:
                break  # Sortir dès qu'on trouve une valeur non survente
        
        return oversold_count
    
    def _get_ema_trend(self, indicators: Dict) -> str:
        """
        Analyse le positionnement des EMA pour déterminer la tendance
        """
        ema_data = indicators.get("ema", {})
        if "ema_9" in ema_data and "ema_21" in ema_data and "ema_50" in ema_data:
            ema_short = ema_data["ema_9"].iloc[-1]
            ema_medium = ema_data["ema_21"].iloc[-1]
            ema_long = ema_data["ema_50"].iloc[-1]
            
            # Alignement parfaitement haussier
            if ema_short > ema_medium > ema_long:
                return "strong_bullish"
            
            # Alignement partiellement haussier
            elif ema_short > ema_medium and ema_short > ema_long:
                return "bullish"
            
            # Alignement parfaitement baissier
            elif ema_short < ema_medium < ema_long:
                return "strong_bearish"
            
            # Alignement partiellement baissier
            elif ema_short < ema_medium and ema_short < ema_long:
                return "bearish"
        
        return "neutral"

    def _get_adx_trend(self, indicators: Dict) -> str:
        """
        Utilise l'ADX pour déterminer la force et la direction de la tendance
        """
        adx_data = indicators.get("adx", {})
        if "adx" in adx_data and "plus_di" in adx_data and "minus_di" in adx_data:
            adx = adx_data["adx"].iloc[-1]
            plus_di = adx_data["plus_di"].iloc[-1]
            minus_di = adx_data["minus_di"].iloc[-1]
            
            # Tendance forte (ADX > 25)
            if adx > 25:
                # Tendance haussière forte
                if plus_di > minus_di and plus_di > 25:
                    return "strong_bullish"
                # Tendance baissière forte
                elif minus_di > plus_di and minus_di > 25:
                    return "strong_bearish"
            
            # Tendance modérée (ADX entre 15 et 25)
            elif adx > 15:
                if plus_di > minus_di:
                    return "bullish"
                elif minus_di > plus_di:
                    return "bearish"
        
        return "neutral"

    def _get_price_slope_trend(self, ohlcv: pd.DataFrame) -> str:
        """
        Analyse la pente des prix récents pour déterminer la tendance
        """
        if len(ohlcv) < 10:
            return "neutral"
        
        # Pente sur 10 périodes
        recent_closes = ohlcv['close'].tail(10).values
        x = np.arange(len(recent_closes))
        slope, _, _, _, _ = np.polyfit(x, recent_closes, 1, full=True)
        
        # Normaliser la pente par rapport au prix moyen
        avg_price = np.mean(recent_closes)
        norm_slope = slope[0] / avg_price * 100
        
        if norm_slope > 1.0:
            return "strong_bullish"
        elif norm_slope > 0.3:
            return "bullish"
        elif norm_slope < -1.0:
            return "strong_bearish"
        elif norm_slope < -0.3:
            return "bearish"
        
        return "neutral"

    def _get_price_structure_trend(self, ohlcv: pd.DataFrame) -> str:
        """
        Analyse des structures de prix pour déterminer la tendance
        """
        if len(ohlcv) < 20:
            return "neutral"
        
        # Identifier les niveaux importants (hauts/bas)
        recent_highs = ohlcv['high'].rolling(5).max()
        recent_lows = ohlcv['low'].rolling(5).min()
        
        # Vérifier si les hauts/bas sont ascendants ou descendants
        higher_highs = recent_highs.iloc[-1] > recent_highs.iloc[-10]
        higher_lows = recent_lows.iloc[-1] > recent_lows.iloc[-10]
        lower_highs = recent_highs.iloc[-1] < recent_highs.iloc[-10]
        lower_lows = recent_lows.iloc[-1] < recent_lows.iloc[-10]
        
        # Structure haussière: hauts plus hauts ET bas plus hauts
        if higher_highs and higher_lows:
            return "strong_bullish"
        
        # Structure partiellement haussière
        elif higher_lows:
            return "bullish"
        
        # Structure baissière: hauts plus bas ET bas plus bas
        elif lower_highs and lower_lows:
            return "strong_bearish"
        
        # Structure partiellement baissière
        elif lower_highs:
            return "bearish"
        
        return "neutral"

    def _get_volume_trend(self, ohlcv: pd.DataFrame) -> str:
        """
        Analyse le volume pour confirmer la tendance des prix
        """
        if len(ohlcv) < 10 or 'volume' not in ohlcv.columns:
            return "neutral"
        
        # Calculer le volume moyen sur 10 périodes
        avg_volume = ohlcv['volume'].rolling(10).mean()
        
        # Comparer les 3 derniers volumes à la moyenne
        recent_vols = ohlcv['volume'].iloc[-3:].values
        recent_prices = ohlcv['close'].iloc[-3:].values
        
        # Prix en hausse + volume en hausse = confirmation haussière
        if recent_prices[-1] > recent_prices[0]:
            # Volume croissant avec les prix
            if recent_vols[-1] > avg_volume.iloc[-1] * 1.2:
                return "strong_bullish"
            elif recent_vols[-1] > avg_volume.iloc[-1]:
                return "bullish"
        
        # Prix en baisse + volume en hausse = confirmation baissière
        elif recent_prices[-1] < recent_prices[0]:
            # Volume croissant avec la baisse des prix
            if recent_vols[-1] > avg_volume.iloc[-1] * 1.2:
                return "strong_bearish"
            elif recent_vols[-1] > avg_volume.iloc[-1]:
                return "bearish"
        
        return "neutral"