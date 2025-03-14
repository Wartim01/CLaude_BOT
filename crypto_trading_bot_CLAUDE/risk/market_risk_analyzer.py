"""
Market risk analyzer for assessing market conditions
"""
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional
import time
from datetime import datetime

from utils.logger import setup_logger
from config.trading_params import MAX_DRAWDOWN_LIMIT, POSITION_SIZING_METHOD  # Example constants

logger = setup_logger("market_risk_analyzer")

class MarketRiskAnalyzer:
    """Analyzes market risk based on various metrics and indicators"""
    
    def __init__(self):
        """Initialize the market risk analyzer"""
        # Risk thresholds
        self.risk_levels = {
            "low": 30,      # Below this is low risk
            "medium": 60,   # Below this is medium risk, above is high risk
            "high": 85      # Above this is extreme risk
        }
        
        # Indicators to use for risk analysis
        self.indicators = [
            "volatility",   # Short-term volatility
            "trend",        # Trend strength
            "volume",       # Volume anomalies
            "correlation",  # Correlation between assets
            "resistance"    # Proximity to key resistance/support
        ]
        
        # Weights for each indicator
        self.weights = {
            "volatility": 0.3,
            "trend": 0.2,
            "volume": 0.2,
            "correlation": 0.15,
            "resistance": 0.15
        }
        
        # Risk assessment history
        self.risk_history = []
        self.last_assessment_time = None
        self.last_risk_score = None
        self.last_risk_level = "unknown"
        
    def calculate_market_risk(self, market_data: Dict[str, pd.DataFrame], 
                           weights: Optional[Dict[str, float]] = None) -> Dict[str, Any]:
        """
        Calculate market risk score based on multiple indicators
        
        Args:
            market_data: Dictionary mapping symbol to DataFrames with market data
            weights: Custom weights for indicators (optional)
            
        Returns:
            Risk assessment dictionary
        """
        # Use default weights if not provided
        weights = weights or self.weights
        
        # Timestamp
        timestamp = int(time.time() * 1000)
        
        # If no data or only minimal data, return high risk by default
        if not market_data or len(market_data) == 0:
            risk_assessment = {
                "risk_score": 75.0,
                "level": "high",
                "reason": "Insufficient market data",
                "timestamp": timestamp,
                "factors": {}
            }
            return risk_assessment
        
        try:
            # 1. Calculate volatility risk
            volatility_risk = self._calculate_volatility_risk(market_data)
            
            # 2. Calculate trend risk
            trend_risk = self._calculate_trend_risk(market_data)
            
            # 3. Calculate volume risk
            volume_risk = self._calculate_volume_risk(market_data)
            
            # 4. Calculate correlation risk
            correlation_risk = self._calculate_correlation_risk(market_data)
            
            # 5. Calculate resistance proximity risk
            resistance_risk = self._calculate_resistance_risk(market_data)
            
            # Combine risk scores using weights
            risk_factors = {
                "volatility": volatility_risk,
                "trend": trend_risk,
                "volume": volume_risk,
                "correlation": correlation_risk,
                "resistance": resistance_risk
            }
            
            # Calculate weighted risk score
            weighted_score = 0
            for factor, score in risk_factors.items():
                weighted_score += score * weights.get(factor, self.weights.get(factor, 0.2))
            
            # Determine risk level
            if weighted_score < self.risk_levels["low"]:
                risk_level = "low"
            elif weighted_score < self.risk_levels["medium"]:
                risk_level = "medium"
            elif weighted_score < self.risk_levels["high"]:
                risk_level = "high"
            else:
                risk_level = "extreme"
            
            # Create risk assessment
            risk_assessment = {
                "risk_score": float(weighted_score),
                "level": risk_level,
                "factors": risk_factors,
                "timestamp": timestamp,
                "symbol_count": len(market_data)
            }
            
            # Store in risk history
            self.risk_history.append({
                "timestamp": timestamp,
                "score": float(weighted_score),
                "level": risk_level
            })
            
            # Trim history to last 100 entries
            if len(self.risk_history) > 100:
                self.risk_history = self.risk_history[-100:]
            
            self.last_assessment_time = timestamp
            
            return risk_assessment
            
        except Exception as e:
            logger.error(f"Error calculating market risk: {str(e)}")
            
            # Return a moderate risk assessment in case of error
            return {
                "risk_score": 50.0,
                "level": "medium",
                "reason": f"Error calculating risk: {str(e)}",
                "timestamp": timestamp,
                "factors": {}
            }
    
    def _calculate_volatility_risk(self, market_data: Dict[str, pd.DataFrame]) -> float:
        """
        Calculate risk based on market volatility
        
        Args:
            market_data: Dictionary of DataFrames with market data
            
        Returns:
            Volatility risk score (0-100)
        """
        # Get average ATR ratio across all symbols
        atr_ratios = []
        
        for symbol, df in market_data.items():
            if 'atr' not in df.columns or 'close' not in df.columns:
                continue
                
            # Calculate ATR as percentage of price
            atr_ratio = df['atr'].iloc[-1] / df['close'].iloc[-1] * 100
            atr_ratios.append(atr_ratio)
        
        if not atr_ratios:
            return 50.0  # Default if no data
        
        # Average ATR ratio
        avg_atr_ratio = np.mean(atr_ratios)
        
        # Convert to risk score (0-100)
        # Typical ATR is 1-3% for crypto, >5% is very volatile
        if avg_atr_ratio < 1.0:
            volatility_risk = 20.0
        elif avg_atr_ratio < 2.0:
            volatility_risk = 40.0
        elif avg_atr_ratio < 3.0:
            volatility_risk = 60.0
        elif avg_atr_ratio < 5.0:
            volatility_risk = 80.0
        else:
            volatility_risk = 100.0
        
        return volatility_risk
    
    def _calculate_trend_risk(self, market_data: Dict[str, pd.DataFrame]) -> float:
        """
        Calculate risk based on market trend strength
        
        Args:
            market_data: Dictionary of DataFrames with market data
            
        Returns:
            Trend risk score (0-100)
        """
        # Count how many symbols are in uptrend, downtrend, or ranging
        uptrend_count = 0
        downtrend_count = 0
        ranging_count = 0
        
        for symbol, df in market_data.items():
            # Skip if necessary columns aren't available
            if not all(col in df.columns for col in ['ema_9', 'ema_21', 'close']):
                continue
            
            # Check trend using EMAs
            ema_9 = df['ema_9'].iloc[-1]
            ema_21 = df['ema_21'].iloc[-1]
            close = df['close'].iloc[-1]
            
            # Uptrend if price above EMAs and 9 EMA above 21 EMA
            if close > ema_9 > ema_21:
                uptrend_count += 1
            # Downtrend if price below EMAs and 9 EMA below 21 EMA
            elif close < ema_9 < ema_21:
                downtrend_count += 1
            else:
                ranging_count += 1
        
        total_count = uptrend_count + downtrend_count + ranging_count
        
        if total_count == 0:
            return 50.0  # Default if no data
        
        # Calculate trend consistency (0-100)
        # Higher value means more consistent trend (whether up or down)
        trend_consistency = 100 * (max(uptrend_count, downtrend_count) / total_count)
        
        # In strong downtrends, risk is higher
        if downtrend_count > uptrend_count:
            trend_risk = 60.0 + (trend_consistency * 0.4)  # 60-100
        else:
            trend_risk = 60.0 - (trend_consistency * 0.4)  # 20-60
        
        return trend_risk
    
    def _calculate_volume_risk(self, market_data: Dict[str, pd.DataFrame]) -> float:
        """
        Calculate risk based on volume anomalies
        
        Args:
            market_data: Dictionary of DataFrames with market data
            
        Returns:
            Volume risk score (0-100)
        """
        # Calculate volume ratios (current volume / average volume)
        volume_ratios = []
        
        for symbol, df in market_data.items():
            if 'volume' not in df.columns:
                continue
                
            # Calculate current volume vs. 20-day average
            if len(df) < 20:
                continue
                
            current_volume = df['volume'].iloc[-1]
            avg_volume = df['volume'].iloc[-20:].mean()
            
            if avg_volume > 0:
                volume_ratio = current_volume / avg_volume
                volume_ratios.append(volume_ratio)
        
        if not volume_ratios:
            return 50.0  # Default if no data
        
        # Average volume ratio
        avg_volume_ratio = np.mean(volume_ratios)
        
        # Convert to risk score (0-100)
        # Very low or very high volume indicates risk
        if avg_volume_ratio < 0.5:
            # Very low volume (potentially illiquid)
            volume_risk = 70.0
        elif avg_volume_ratio < 0.8:
            # Below average volume
            volume_risk = 60.0
        elif avg_volume_ratio < 1.5:
            # Normal volume
            volume_risk = 40.0
        elif avg_volume_ratio < 3.0:
            # High volume (could be volatile)
            volume_risk = 60.0
        else:
            # Very high volume (potential market turning point)
            volume_risk = 80.0
        
        return volume_risk
    
    def _calculate_correlation_risk(self, market_data: Dict[str, pd.DataFrame]) -> float:
        """
        Calculate risk based on correlation between assets
        
        Args:
            market_data: Dictionary of DataFrames with market data
            
        Returns:
            Correlation risk score (0-100)
        """
        if len(market_data) < 2:
            return 50.0  # Default if not enough data
        
        # Extract recent returns for each symbol
        returns_data = {}
        
        for symbol, df in market_data.items():
            if 'close' not in df.columns or len(df) < 10:
                continue
                
            # Calculate returns
            returns = df['close'].pct_change().iloc[-10:].dropna()
            if not returns.empty:
                returns_data[symbol] = returns
        
        if len(returns_data) < 2:
            return 50.0  # Default if not enough data
        
        # Calculate correlation matrix
        correlations = []
        symbols = list(returns_data.keys())
        
        for i in range(len(symbols)):
            for j in range(i+1, len(symbols)):
                sym1 = symbols[i]
                sym2 = symbols[j]
                
                # Align the series by index
                joined = pd.concat([returns_data[sym1], returns_data[sym2]], axis=1).dropna()

    def analyze_risk(self, market_data: dict) -> dict:
        """
        Analyse le risque global du marché en fonction de la volatilité,
        de la corrélation inter-actifs et d'autres facteurs.
        
        Args:
            market_data: Données de marché nécessaires au calcul.
            
        Returns:
            Dictionnaire contenant risk_score, risk_level, et d'autres détails.
        """
        try:
            # ...existing code to extract factors from market_data...
            # Exemple simplifié de calcul du risk_score:
            volatility = market_data.get("volatility", 0.0)  # Ex: mesure en pourcentage
            correlation = market_data.get("average_correlation", 0.0)  # Ex: valeur entre 0 et 1
            
            # Calculer un score combiné (poids arbitraires)
            risk_score = volatility * 0.7 + correlation * 0.3
            
            # Définir le niveau de risque selon des seuils prédéfinis
            if risk_score < 20:
                risk_level = "low"
            elif risk_score < 40:
                risk_level = "medium"
            elif risk_score < 60:
                risk_level = "high"
            else:
                risk_level = "extreme"
            
            # Stocker le dernier résultat
            self.last_risk_score = risk_score
            self.last_risk_level = risk_level
            
            return {
                "risk_score": risk_score,
                "risk_level": risk_level,
                "details": {
                    "volatility": volatility,
                    "average_correlation": correlation
                }
            }
        except Exception as e:
            # En cas d'erreur, renvoyer le dernier risque connu
            # et signaler l'erreur dans details
            return {
                "risk_score": self.last_risk_score if self.last_risk_score is not None else 0,
                "risk_level": self.last_risk_level,
                "error": str(e)
            }