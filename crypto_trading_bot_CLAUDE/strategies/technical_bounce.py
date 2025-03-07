"""
Technical Bounce Strategy
A strategy that looks for technical rebounds from support/resistance levels
"""
import numpy as np
import pandas as pd
from typing import Dict, Optional

class TechnicalBounceStrategy:
    """
    Trading strategy that identifies potential bounce opportunities
    from oversold conditions and technical support levels
    """
    
    def __init__(self, data_fetcher, market_analyzer, scoring_engine):
        """
        Initialize the strategy with required components
        
        Args:
            data_fetcher: Component for fetching market data
            market_analyzer: Component for analyzing market conditions
            scoring_engine: Component for scoring trade opportunities
        """
        self.data_fetcher = data_fetcher
        self.market_analyzer = market_analyzer
        self.scoring_engine = scoring_engine
        
        # Strategy parameters
        self.rsi_oversold = 30
        self.rsi_overbought = 70
        self.min_adx = 20  # Minimum ADX for trend strength
        self.bb_threshold = 0.05  # How close to BB lower band (in %)
    
    def find_trading_opportunity(self, symbol: str) -> Optional[Dict]:
        """
        Look for trading opportunities based on technical bounce conditions
        
        Args:
            symbol: Trading pair symbol (e.g., "BTCUSDT")
            
        Returns:
            Dictionary with opportunity details if found, None otherwise
        """
        # Get market data with indicators
        market_data = self.data_fetcher.get_market_data(symbol)
        
        if market_data is None or "primary_timeframe" not in market_data:
            return None
            
        # Extract indicators
        indicators = market_data["primary_timeframe"]["indicators"]
        ohlcv = market_data["primary_timeframe"]["ohlcv"]
        
        if not indicators or ohlcv.empty:
            return None
            
        # Check if market conditions are favorable
        market_state = self.market_analyzer.analyze_market_state(symbol)
        if not market_state.get("favorable", False):
            return None
            
        # Check for oversold conditions (RSI)
        rsi = indicators.get("rsi")
        if rsi is None or len(rsi) < 2:
            return None
            
        current_rsi = rsi.iloc[-1]
        prev_rsi = rsi.iloc[-2]
        
        # Check for potential bounce from oversold
        rsi_bounce = current_rsi > prev_rsi and prev_rsi < self.rsi_oversold
        
        # Check for price near Bollinger lower band
        bb = indicators.get("bollinger", {})
        if not bb or "lower" not in bb or "percent_b" not in bb:
            return None
            
        # Bollinger bounce condition
        bb_percent = bb["percent_b"].iloc[-1]
        bb_bounce = bb_percent < self.bb_threshold and bb_percent > 0
        
        # Check volume spike
        volume_data = self.data_fetcher.detect_volume_spike(symbol)
        volume_spike = volume_data.get("spike", False) and volume_data.get("bullish", False)
        
        # Trend strength filter
        adx = indicators.get("adx", {})
        adx_value = adx["adx"].iloc[-1] if isinstance(adx, dict) and "adx" in adx else 0
        trend_strength = adx_value > self.min_adx
        
        # Overall bounce condition
        bounce_condition = (rsi_bounce or bb_bounce) and (trend_strength or volume_spike)
        
        if not bounce_condition:
            return None
            
        # Calculate opportunity score
        factors = {
            "rsi_level": current_rsi / 100,  # 0-1 scale
            "bb_position": bb_percent,       # 0-1 scale
            "trend_strength": adx_value / 50 if adx_value < 50 else 1.0,
            "volume_strength": volume_data.get("ratio", 1.0) / 3 if volume_spike else 0.3
        }
        
        # Get score from scoring engine
        score = self.scoring_engine.calculate_opportunity_score(factors)
        
        # Create opportunity details
        opportunity = {
            "symbol": symbol,
            "type": "technical_bounce",
            "direction": "buy",
            "score": score,
            "timestamp": pd.Timestamp.now().isoformat(),
            "factors": factors,
            "indicators": {
                "rsi": float(current_rsi),
                "bb_percent": float(bb_percent),
                "adx": float(adx_value) if adx_value else 0,
                "volume_spike": volume_spike
            }
        }
        
        return opportunity