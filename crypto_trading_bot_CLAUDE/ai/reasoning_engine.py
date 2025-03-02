# ai/reasoning_engine.py
"""
Moteur de raisonnement pour expliquer les décisions et générer du texte
"""
import logging
from typing import Dict, List, Optional, Union
from datetime import datetime

from utils.logger import setup_logger

logger = setup_logger("reasoning_engine")

class ReasoningEngine:
    """
    Génère des explications textuelles pour les décisions de trading
    """
    def __init__(self):
        self.templates = self._initialize_templates()
    
    def _initialize_templates(self) -> Dict:
        """
        Initialise les templates pour la génération de texte
        
        Returns:
            Dictionnaire de templates
        """
        return {
            "technical_bounce": {
                "opportunity": (
                    "Opportunité de rebond technique détectée sur {symbol} avec un score de {score}/100. "
                    "Le prix est actuellement à {price} {base_currency}, montrant des signes de retournement haussier "
                    "après une période de baisse. {signals_text} "
                    "Le stop-loss est placé à {stop_loss} (-{stop_loss_percent}%) et "
                    "le take-profit à {take_profit} (+{take_profit_percent}%), "
                    "donnant un ratio risque/récompense de {risk_reward_ratio:.2f}."
                ),
                "market_conditions": (
                    "Conditions de marché: {market_conditions}. "
                    "RSI actuel: {rsi:.1f}. "
                    "Force de tendance (ADX): {adx:.1f}. "
                    "Volatilité (ATR): {atr:.2f}."
                ),
                "entry_reasoning": (
                    "Raisonnement d'entrée: "
                    "Le prix est {price_position} avec {candle_pattern}. "
                    "{volume_analysis} "
                    "{divergence_analysis} "
                    "Les signaux techniques indiquent une forte probabilité de rebond à court terme."
                )
            },
            "trade_result": {
                "success": (
                    "Trade sur {symbol} clôturé avec profit: +{pnl_percent:.2f}% (+{pnl_absolute:.2f} {currency}). "
                    "Durée du trade: {duration}. "
                    "Entrée à {entry_price}, sortie à {exit_price}. "
                    "Raison de sortie: {exit_reason}."
                ),
                "failure": (
                    "Trade sur {symbol} clôturé avec perte: {pnl_percent:.2f}% ({pnl_absolute:.2f} {currency}). "
                    "Durée du trade: {duration}. "
                    "Entrée à {entry_price}, sortie à {exit_price}. "
                    "Raison de sortie: {exit_reason}. "
                    "Leçon à retenir: {lesson}."
                )
            }
        }
    
    def generate_opportunity_explanation(self, opportunity: Dict) -> str:
        """
        Génère une explication détaillée pour une opportunité de trading
        
        Args:
            opportunity: Données de l'opportunité
            
        Returns:
            Explication textuelle
        """
        strategy = opportunity.get("strategy", "unknown")
        
        if strategy not in self.templates:
            return f"Opportunité de trading détectée avec la stratégie '{strategy}'."
        
        template = self.templates[strategy]
        
        # Extraire les données de l'opportunité
        symbol = opportunity.get("symbol", "UNKNOWN")
        score = opportunity.get("score", 0)
        entry_price = opportunity.get("entry_price", 0)
        stop_loss = opportunity.get("stop_loss", 0)
        take_profit = opportunity.get("take_profit", 0)
        signals = opportunity.get("signals", {}).get("signals", [])
        
        # Calculer les pourcentages
        stop_loss_percent = abs((stop_loss - entry_price) / entry_price * 100)
        take_profit_percent = abs((take_profit - entry_price) / entry_price * 100)
        risk_reward_ratio = take_profit_percent / stop_loss_percent if stop_loss_percent > 0 else 0
        
        # Extraire la devise de base
        base_currency = "USDT"  # Par défaut
        if symbol.endswith("USDT"):
            base_currency = "USDT"
        
        # Formater les signaux
        signals_text = "Signaux détectés: " + ", ".join(signals) + "." if signals else ""
        
        # Section 1: Opportunité de base
        explanation = template["opportunity"].format(
            symbol=symbol,
            score=score,
            price=entry_price,
            base_currency=base_currency,
            signals_text=signals_text,
            stop_loss=stop_loss,
            stop_loss_percent=f"{stop_loss_percent:.2f}",
            take_profit=take_profit,
            take_profit_percent=f"{take_profit_percent:.2f}",
            risk_reward_ratio=risk_reward_ratio
        )
        
        # Section 2: Conditions de marché
        market_conditions = opportunity.get("market_conditions", {})
        market_conditions_text = "normales"
        
        if market_conditions.get("details"):
            details = market_conditions.get("details", {})
            
            if details.get("adx", {}).get("strong_trend", False):
                if details.get("adx", {}).get("bearish_trend", False):
                    market_conditions_text = "tendance baissière forte"
                else:
                    market_conditions_text = "tendance haussière forte"
            elif details.get("bollinger", {}).get("high_volatility", False):
                market_conditions_text = "haute volatilité"
            else:
                market_conditions_text = "favorables pour un rebond"
        
        # Extraire les indicateurs
        indicators = opportunity.get("indicators", {})
        rsi = indicators.get("rsi", 50)
        adx = market_conditions.get("details", {}).get("adx", {}).get("value", 25)
        atr = indicators.get("atr", 0.01)
        
        explanation += " " + template["market_conditions"].format(
            market_conditions=market_conditions_text,
            rsi=rsi,
            adx=adx,
            atr=atr
        )
        
        # Section 3: Raisonnement d'entrée
        price_position = "sous la bande inférieure de Bollinger" if indicators.get("bollinger", {}).get("percent_b", 0.5) < 0 else "proche d'un support technique"
        
        candle_pattern = "une bougie de retournement"
        if "Mèche inférieure significative" in signals:
            candle_pattern = "une mèche inférieure significative indiquant un rejet des prix bas"
        elif "Chandelier haussier après chandelier baissier" in signals:
            candle_pattern = "un chandelier haussier après une série de chandeliers baissiers"
        
        volume_analysis = "Le volume est normal."
        if "Pic de volume haussier" in signals:
            volume_analysis = "Un pic de volume haussier a été détecté, indiquant un fort intérêt acheteur."
        
        divergence_analysis = ""
        if "Divergence haussière RSI détectée" in signals:
            divergence_analysis = "Une divergence haussière a été détectée entre le prix et le RSI, un signal fort de retournement. "
        
        explanation += " " + template["entry_reasoning"].format(
            price_position=price_position,
            candle_pattern=candle_pattern,
            volume_analysis=volume_analysis,
            divergence_analysis=divergence_analysis
        )
        
        return explanation
    
    def generate_trade_result_explanation(self, trade_result: Dict) -> str:
        """
        Génère une explication pour le résultat d'un trade
        
        Args:
            trade_result: Résultat du trade
            
        Returns:
            Explication textuelle
        """
        # Déterminer si c'est un succès ou un échec
        pnl_percent = trade_result.get("pnl_percent", 0)
        is_success = pnl_percent > 0
        
        template_key = "success" if is_success else "failure"
        template = self.templates["trade_result"][template_key]
        
        # Extraire les données du trade
        symbol = trade_result.get("symbol", "UNKNOWN")
        pnl_absolute = trade_result.get("pnl_absolute", 0)
        entry_price = trade_result.get("entry_price", 0)
        exit_price = trade_result.get("exit_price", 0)
        exit_reason = trade_result.get("exit_reason", "Take-profit/Stop-loss")
        currency = "USDT"
        
        # Calculer la durée du trade
        entry_time = trade_result.get("entry_time")
        close_time = trade_result.get("close_time")
        
        if entry_time and close_time:
            if isinstance(entry_time, str):
                entry_time = datetime.fromisoformat(entry_time)
            if isinstance(close_time, str):
                close_time = datetime.fromisoformat(close_time)
            
            duration_seconds = (close_time - entry_time).total_seconds()
            
            if duration_seconds < 60:
                duration = f"{int(duration_seconds)} secondes"
            elif duration_seconds < 3600:
                duration = f"{int(duration_seconds/60)} minutes"
            else:
                duration = f"{duration_seconds/3600:.1f} heures"
        else:
            duration = "inconnue"
        
        # Générer la leçon à retenir pour les trades en échec
        lesson = ""
        if not is_success:
            if pnl_percent > -2:
                lesson = "La perte est minime, la stratégie reste valide"
            elif "stop_loss" in exit_reason.lower():
                lesson = "Revoir les critères d'entrée et les niveaux de stop-loss"
            else:
                lesson = "Analyser les signaux contradictoires et la vitesse de retournement du marché"
        
        # Formater l'explication
        explanation = template.format(
            symbol=symbol,
            pnl_percent=pnl_percent,
            pnl_absolute=pnl_absolute,
            currency=currency,
            duration=duration,
            entry_price=entry_price,
            exit_price=exit_price,
            exit_reason=exit_reason,
            lesson=lesson
        )
        
        return explanation