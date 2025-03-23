"""
Technical Bounce Strategy
A strategy that looks for technical rebounds from support/resistance levels
"""
import os
import json
import numpy as np
import pandas as pd
from typing import Dict, Optional
from datetime import datetime, timedelta

from utils.logger import setup_logger
from config.trading_params import MINIMUM_SCORE_TO_TRADE, RSI_OVERSOLD, BB_PERIOD
from config.config import DATA_DIR

logger = setup_logger("technical_bounce")

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
        
        # Get current minimum score to trade
        self.min_score = MINIMUM_SCORE_TO_TRADE
        
        # Storage for missed opportunities tracking
        self.opportunities_history_dir = os.path.join(DATA_DIR, "opportunities")
        os.makedirs(self.opportunities_history_dir, exist_ok=True)
        self.opportunities_file = os.path.join(self.opportunities_history_dir, "missed_opportunities.json")
        self.loaded_opportunities = False
        self.missed_opportunities = []
        self.detected_opportunities = []
        self._load_opportunities_data()
        
        logger.info(f"Technical Bounce Strategy initialized with min_score={self.min_score}")
        
    def _load_opportunities_data(self):
        """Load stored opportunities data from file"""
        try:
            if os.path.exists(self.opportunities_file):
                with open(self.opportunities_file, 'r') as f:
                    data = json.load(f)
                    self.missed_opportunities = data.get('missed', [])
                    self.detected_opportunities = data.get('detected', [])
                    self.loaded_opportunities = True
                    logger.debug(f"Loaded {len(self.missed_opportunities)} missed and {len(self.detected_opportunities)} detected opportunities")
            else:
                logger.debug("No opportunities history file found, starting fresh")
                self.missed_opportunities = []
                self.detected_opportunities = []
                self.loaded_opportunities = True
        except Exception as e:
            logger.error(f"Error loading opportunities data: {str(e)}")
            # Initialize with empty lists as fallback
            self.missed_opportunities = []
            self.detected_opportunities = []
            self.loaded_opportunities = True

    def _save_opportunities_data(self):
        """Save opportunities data to file"""
        try:
            # Prune old entries to keep file size manageable (keep last 500)
            if len(self.missed_opportunities) > 500:
                self.missed_opportunities = self.missed_opportunities[-500:]
            if len(self.detected_opportunities) > 500:
                self.detected_opportunities = self.detected_opportunities[-500:]
                
            # Prepare data structure
            data = {
                'missed': self.missed_opportunities,
                'detected': self.detected_opportunities,
                'last_updated': datetime.now().isoformat()
            }
            
            with open(self.opportunities_file, 'w') as f:
                json.dump(data, f, indent=2, default=str)
                
        except Exception as e:
            logger.error(f"Error saving opportunities data: {str(e)}")

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
        
        # Track this opportunity for threshold adjustment analysis
        self._record_opportunity(symbol, opportunity)
        
        # Return only if it meets minimum threshold
        if score >= self.min_score:
            return opportunity
        else:
            # Record as missed opportunity
            self._record_missed_opportunity(symbol, opportunity)
            return None
            
    def _record_opportunity(self, symbol: str, opportunity: Dict):
        """Record a detected opportunity regardless of execution"""
        if not self.loaded_opportunities:
            self.loaded_opportunities = True  # first time loading done
            
        # Add to detected opportunities list
        detection = {
            "symbol": symbol,
            "score": opportunity["score"],
            "timestamp": datetime.now().isoformat(),
            "min_score_at_detection": self.min_score,
            "direction": opportunity.get("direction", "unknown")
        }
        
        self.detected_opportunities.append(detection)
        
        # Save to disk periodically (every 10 detections)
        if len(self.detected_opportunities) % 10 == 0:
            self._save_opportunities_data()
            
    def _record_missed_opportunity(self, symbol: str, opportunity: Dict) -> None:
        """Record details of opportunities that didn't meet the threshold"""
        opportunity_time = datetime.now()
        missed = {
            "symbol": symbol,
            "timestamp": opportunity_time,
            "score": opportunity.get("score", 0),
            "min_score": self.min_score,
            "difference": self.min_score - opportunity.get("score", 0)
        }
        
        self.missed_opportunities.append(missed)
        logger.debug(f"Missed opportunity for {symbol}: score {opportunity.get('score', 0)} vs threshold {self.min_score}")
        
        # Keep only recent missed opportunities (last 24 hours)
        cutoff_time = opportunity_time - timedelta(hours=24)
        self.missed_opportunities = [m for m in self.missed_opportunities 
                                   if m["timestamp"] > cutoff_time]

    def get_missed_opportunities_analysis(self, hours: int = 6) -> Dict:
        """
        Analyze missed opportunities over the specified period
        
        Args:
            hours: Number of hours to look back
            
        Returns:
            Dictionary with analysis results
        """
        cutoff_time = datetime.now() - timedelta(hours=hours)
        recent_misses = [m for m in self.missed_opportunities 
                         if m["timestamp"] > cutoff_time]
        
        if not recent_misses:
            return {
                "count": 0,
                "period_hours": hours,
                "avg_score": 0,
                "near_misses": 0,  # Opportunities that were close to threshold
                "has_clustering": False
            }
        
        # Calculate statistics
        avg_score = sum(m["score"] for m in recent_misses) / len(recent_misses)
        
        # Near misses - opportunities within 5 points of threshold
        near_misses = [m for m in recent_misses 
                      if self.min_score - m["score"] <= 5.0]
        
        # Look for clustering (many opportunities in short timeframe)
        timestamps = [m["timestamp"] for m in recent_misses]
        timestamps.sort()
        
        has_clustering = False
        if len(timestamps) >= 3:
            for i in range(len(timestamps) - 2):
                # If 3 or more opportunities within 1 hour
                if (timestamps[i+2] - timestamps[i]).total_seconds() < 3600:
                    has_clustering = True
                    break
        
        return {
            "count": len(recent_misses),
            "period_hours": hours,
            "avg_score": avg_score,
            "near_misses": len(near_misses),
            "has_clustering": has_clustering
        }

    def get_opportunities_stats(self, lookback_hours: int = 24) -> Dict:
        """Get statistics about detected and missed opportunities"""
        if not self.loaded_opportunities:
            self._load_opportunities_data()
            
        now = datetime.now()
        lookback_time = now - timedelta(hours=lookback_hours)
        
        # Filter opportunities within lookback period
        recent_detected = [
            op for op in self.detected_opportunities
            if datetime.fromisoformat(op["timestamp"]) > lookback_time
        ]
        
        recent_missed = [
            op for op in self.missed_opportunities
            if datetime.fromisoformat(op["timestamp"]) > lookback_time
        ]
        
        # Calculate statistics
        total_opportunities = len(recent_detected)
        missed_opportunities = len(recent_missed)
        executed_opportunities = total_opportunities - missed_opportunities
        
        # Calculate average scores
        avg_detected_score = np.mean([op["score"] for op in recent_detected]) if recent_detected else 0
        avg_missed_score = np.mean([op["score"] for op in recent_missed]) if recent_missed else 0
        
        # Calculate near-miss opportunities (within 5 points of threshold)
        near_misses = [
            op for op in recent_missed
            if op["threshold_gap"] <= 5
        ]
        
        return {
            "lookback_hours": lookback_hours,
            "total_opportunities": total_opportunities,
            "executed_opportunities": executed_opportunities,
            "missed_opportunities": missed_opportunities,
            "near_misses": len(near_misses),
            "avg_detected_score": avg_detected_score,
            "avg_missed_score": avg_missed_score,
            "current_threshold": self.min_score
        }
        
    def update_min_score(self, new_score: float) -> None:
        """Update the minimum score threshold for trading"""
        old_score = self.min_score
        self.min_score = new_score
        logger.info(f"Updated minimum score threshold from {old_score} to {new_score}")