"""
Module for detecting market anomalies and extreme events
"""
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Union, Any
from datetime import datetime, timedelta

from utils.logger import setup_logger

logger = setup_logger("market_anomaly_detector")

class MarketAnomalyDetector:
    """
    Detects anomalies in market data that might warrant trading caution
    """
    def __init__(self, data_manager, config=None):
        """
        Initialize the market anomaly detector
        
        Args:
            data_manager: Data manager instance for fetching market data
            config: Configuration dictionary for anomaly detection
        """
        self.data_manager = data_manager
        self.config = config or {}
        
        # Default thresholds for anomaly detection
        self.price_change_threshold = self.config.get("price_change_threshold", 5.0)  # 5% change in a short time
        self.volume_spike_threshold = self.config.get("volume_spike_threshold", 3.0)  # 3x average volume
        self.volatility_threshold = self.config.get("volatility_threshold", 2.5)     # 2.5x average volatility
        
        logger.info("Market anomaly detector initialized")
        
    def detect_anomalies(self, symbol: str, timeframe: str) -> Dict[str, Any]:
        """
        Detect market anomalies for a given symbol and timeframe
        
        Args:
            symbol: Trading pair symbol
            timeframe: Timeframe for analysis
            
        Returns:
            Dictionary with anomaly detection results
        """
        try:
            # Get recent market data for analysis
            market_data = self.data_manager.get_recent_data(symbol, timeframe, limit=100)
            if market_data is None or len(market_data) < 30:
                return {"is_anomaly": False, "reason": "Insufficient data for analysis"}
                
            # Check for price anomalies
            price_anomaly = self._detect_price_anomalies(market_data)
            
            # Check for volume anomalies
            volume_anomaly = self._detect_volume_anomalies(market_data)
            
            # Check for volatility anomalies
            volatility_anomaly = self._detect_volatility_anomalies(market_data)
            
            # Combine results
            is_anomaly = price_anomaly["is_anomaly"] or volume_anomaly["is_anomaly"] or volatility_anomaly["is_anomaly"]
            
            # Build description
            description = None
            if is_anomaly:
                anomaly_reasons = []
                if price_anomaly["is_anomaly"]:
                    anomaly_reasons.append(price_anomaly["description"])
                if volume_anomaly["is_anomaly"]:
                    anomaly_reasons.append(volume_anomaly["description"])
                if volatility_anomaly["is_anomaly"]:
                    anomaly_reasons.append(volatility_anomaly["description"])
                    
                description = " AND ".join(anomaly_reasons)
            
            return {
                "is_anomaly": is_anomaly,
                "description": description,
                "symbol": symbol,
                "timeframe": timeframe,
                "price_anomaly": price_anomaly,
                "volume_anomaly": volume_anomaly,
                "volatility_anomaly": volatility_anomaly,
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error detecting anomalies for {symbol} {timeframe}: {str(e)}")
            return {"is_anomaly": False, "reason": f"Error in detection: {str(e)}"}
    
    def _detect_price_anomalies(self, market_data: pd.DataFrame) -> Dict:
        """Detect abnormal price movements"""
        try:
            # Calculate recent price changes
            close_prices = market_data['close'].values
            
            # Short-term change (last 5 candles)
            short_term_change = (close_prices[-1] / close_prices[-5] - 1) * 100
            
            # Medium-term change (last 20 candles)
            medium_term_change = (close_prices[-1] / close_prices[-20] - 1) * 100
            
            # Check if changes exceed thresholds
            is_short_term_anomaly = abs(short_term_change) > self.price_change_threshold
            is_medium_term_anomaly = abs(medium_term_change) > self.price_change_threshold * 1.5
            
            is_anomaly = is_short_term_anomaly or is_medium_term_anomaly
            
            description = None
            if is_anomaly:
                direction = "increase" if short_term_change > 0 or medium_term_change > 0 else "decrease"
                if is_short_term_anomaly:
                    description = f"Abnormal {abs(short_term_change):.2f}% {direction} in price over short period"
                else:
                    description = f"Abnormal {abs(medium_term_change):.2f}% {direction} in price over medium period"
            
            return {
                "is_anomaly": is_anomaly,
                "description": description,
                "short_term_change": short_term_change,
                "medium_term_change": medium_term_change
            }
            
        except Exception as e:
            logger.error(f"Error in price anomaly detection: {str(e)}")
            return {"is_anomaly": False}
    
    def _detect_volume_anomalies(self, market_data: pd.DataFrame) -> Dict:
        """Detect abnormal trading volume"""
        try:
            # Calculate volume statistics
            volumes = market_data['volume'].values
            
            # Calculate average volume (excluding most recent candles)
            avg_volume = np.mean(volumes[-30:-5])
            current_volume = np.mean(volumes[-5:])  # Average of last 5 candles
            
            # Check for volume spike
            volume_ratio = current_volume / avg_volume if avg_volume > 0 else 1.0
            is_anomaly = volume_ratio > self.volume_spike_threshold
            
            description = None
            if is_anomaly:
                description = f"Volume spike detected ({volume_ratio:.2f}x normal volume)"
            
            return {
                "is_anomaly": is_anomaly,
                "description": description,
                "volume_ratio": volume_ratio
            }
            
        except Exception as e:
            logger.error(f"Error in volume anomaly detection: {str(e)}")
            return {"is_anomaly": False}
    
    def _detect_volatility_anomalies(self, market_data: pd.DataFrame) -> Dict:
        """Detect abnormal market volatility"""
        try:
            # Calculate true range as a measure of volatility
            high = market_data['high'].values
            low = market_data['low'].values
            close = market_data['close'].values
            
            # True Range calculation
            tr = []
            for i in range(1, len(close)):
                hl = high[i] - low[i]
                hc = abs(high[i] - close[i-1])
                lc = abs(low[i] - close[i-1])
                tr.append(max(hl, hc, lc))
            
            # Calculate average true range (excluding most recent)
            avg_tr = np.mean(tr[:-5])
            current_tr = np.mean(tr[-5:])  # Average of last 5 candles
            
            # Check for volatility spike
            volatility_ratio = current_tr / avg_tr if avg_tr > 0 else 1.0
            is_anomaly = volatility_ratio > self.volatility_threshold
            
            description = None
            if is_anomaly:
                description = f"Abnormal volatility detected ({volatility_ratio:.2f}x normal levels)"
            
            return {
                "is_anomaly": is_anomaly,
                "description": description,
                "volatility_ratio": volatility_ratio
            }
            
        except Exception as e:
            logger.error(f"Error in volatility anomaly detection: {str(e)}")
            return {"is_anomaly": False}