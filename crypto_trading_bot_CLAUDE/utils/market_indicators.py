"""
Module for market-wide indicators and cross-market analysis to identify
broader market conditions and regime changes
"""
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Union
from datetime import datetime, timedelta
import talib

from utils.logger import setup_logger
from utils.technical_analysis import TechnicalAnalysis

logger = setup_logger("market_indicators")

class MarketIndicators:
    """
    Class for analyzing market-wide indicators and cross-market relationships
    to identify market regimes and conditions
    """
    def __init__(self):
        """Initialize the market indicators analyzer"""
        self.market_regimes = ["risk_on", "risk_off", "neutral", "high_volatility", "low_volatility"]
        self.correlation_threshold = 0.7  # Strong correlation threshold
        self.volatility_threshold = 2.0   # High volatility threshold (multiplier of normal)
        
    def analyze_market_regime(self, market_data: Dict[str, pd.DataFrame], 
                              base_symbol: str = "BTCUSDT",
                              lookback_period: int = 30) -> Dict:
        """
        Analyzes multiple market data to determine the current market regime
        
        Args:
            market_data: Dictionary of DataFrames with OHLCV data for different markets
            base_symbol: Symbol to use as the base for comparisons
            lookback_period: Period to analyze for regime detection
            
        Returns:
            Dictionary with market regime analysis
        """
        result = {
            "regime": "neutral",
            "confidence": 0.0,
            "volatility": "normal",
            "correlations": {},
            "dominant_factors": [],
            "timestamp": datetime.now().isoformat()
        }
        
        try:
            # Check if we have enough data
            if base_symbol not in market_data or market_data[base_symbol].empty:
                result["error"] = f"Base symbol {base_symbol} not found in market data"
                return result
                
            # 1. Calculate volatility across markets
            volatility_scores = {}
            for symbol, df in market_data.items():
                if len(df) < lookback_period:
                    continue
                
                # Calculate historical volatility (annualized standard deviation of returns)
                returns = df['close'].pct_change().dropna()
                if len(returns) < lookback_period:
                    continue
                    
                current_vol = returns.iloc[-lookback_period:].std() * np.sqrt(365)
                historical_vol = returns.iloc[:-lookback_period].std() * np.sqrt(365)
                
                if historical_vol > 0:
                    vol_ratio = current_vol / historical_vol
                else:
                    vol_ratio = 1.0
                
                volatility_scores[symbol] = vol_ratio
            
            # 2. Calculate correlations between base symbol and others
            correlations = {}
            base_returns = market_data[base_symbol]['close'].pct_change().dropna()
            
            for symbol, df in market_data.items():
                if symbol == base_symbol or len(df) < lookback_period:
                    continue
                
                symbol_returns = df['close'].pct_change().dropna()
                
                # Ensure we're using the same timeframe for correlation
                common_index = base_returns.index.intersection(symbol_returns.index)
                if len(common_index) < lookback_period:
                    continue
                
                # Calculate correlation
                correlation = base_returns.loc[common_index[-lookback_period:]].corr(
                    symbol_returns.loc[common_index[-lookback_period:]]
                )
                
                correlations[symbol] = correlation
            
            # 3. Analyze market indicators for the base symbol
            base_data = market_data[base_symbol].iloc[-lookback_period:]
            
            # Calculate moving averages
            base_data['sma_50'] = talib.SMA(base_data['close'].values, timeperiod=min(50, len(base_data)))
            base_data['sma_200'] = talib.SMA(base_data['close'].values, timeperiod=min(200, len(base_data)))
            
            # Check if price is above/below moving averages
            price_above_50ma = base_data['close'].iloc[-1] > base_data['sma_50'].iloc[-1]
            price_above_200ma = base_data['close'].iloc[-1] > base_data['sma_200'].iloc[-1]
            
            # Calculate RSI
            base_data['rsi'] = talib.RSI(base_data['close'].values, timeperiod=14)
            current_rsi = base_data['rsi'].iloc[-1]
            
            # 4. Determine market regime
            # Calculate average volatility
            avg_volatility = np.mean(list(volatility_scores.values()))
            
            # Determine if we're in a high volatility regime
            high_volatility = avg_volatility > self.volatility_threshold
            
            # Count positive and negative correlations
            positive_corr = sum(1 for c in correlations.values() if c > self.correlation_threshold)
            negative_corr = sum(1 for c in correlations.values() if c < -self.correlation_threshold)
            total_corr = len(correlations)
            
            # Determine regime based on collected indicators
            if high_volatility:
                regime = "high_volatility"
                confidence = min(1.0, (avg_volatility - 1.0) / self.volatility_threshold)
                dominant_factors = ["volatility"]
                
                # Check if it's risk-on or risk-off during high volatility
                if price_above_50ma and price_above_200ma and current_rsi > 50:
                    regime = "risk_on"
                    dominant_factors.append("bullish_trend")
                elif not price_above_50ma and not price_above_200ma and current_rsi < 50:
                    regime = "risk_off"
                    dominant_factors.append("bearish_trend")
            
            # If not high volatility, check for risk-on/risk-off based on correlations and trends
            elif price_above_50ma and price_above_200ma and current_rsi > 60:
                regime = "risk_on"
                confidence = min(1.0, (current_rsi - 50) / 30)
                dominant_factors = ["bullish_trend", "positive_momentum"]
            
            elif not price_above_50ma and not price_above_200ma and current_rsi < 40:
                regime = "risk_off"
                confidence = min(1.0, (50 - current_rsi) / 30)
                dominant_factors = ["bearish_trend", "negative_momentum"]
            
            # Check for low volatility regime
            elif avg_volatility < 0.7:  # Significantly lower than normal
                regime = "low_volatility"
                confidence = min(1.0, (1.0 - avg_volatility) / 0.3)
                dominant_factors = ["compressed_volatility"]
            
            else:
                regime = "neutral"
                confidence = 0.5
                dominant_factors = ["mixed_signals"]
            
            # 5. Prepare result
            result["regime"] = regime
            result["confidence"] = float(confidence)
            result["volatility"] = "high" if high_volatility else "low" if avg_volatility < 0.7 else "normal"
            result["correlations"] = {k: float(v) for k, v in correlations.items()}
            result["volatility_scores"] = {k: float(v) for k, v in volatility_scores.items()}
            result["average_volatility"] = float(avg_volatility)
            result["dominant_factors"] = dominant_factors
            
            # Additional technical indicators
            result["technical_indicators"] = {
                "price_above_50ma": bool(price_above_50ma),
                "price_above_200ma": bool(price_above_200ma),
                "rsi": float(current_rsi)
            }
            
        except Exception as e:
            logger.error(f"Error in market regime analysis: {str(e)}")
            result["error"] = f"Error in analysis: {str(e)}"
        
        return result
    
    def detect_regime_change(self, current_regime: Dict, previous_regime: Dict) -> Dict:
        """
        Detects if there has been a change in market regime
        
        Args:
            current_regime: Current market regime analysis
            previous_regime: Previous market regime analysis
            
        Returns:
            Dictionary with regime change analysis
        """
        if not current_regime or not previous_regime:
            return {"change_detected": False, "reason": "Insufficient data"}
        
        # Check if the regime has changed
        regime_changed = current_regime.get("regime") != previous_regime.get("regime")
        
        # Check for significant volatility changes
        volatility_change = False
        current_vol = current_regime.get("average_volatility", 1.0)
        previous_vol = previous_regime.get("average_volatility", 1.0)
        
        if previous_vol > 0:
            vol_change_pct = (current_vol - previous_vol) / previous_vol
            volatility_change = abs(vol_change_pct) > 0.3  # 30% change in volatility
        
        # Check for correlation breakdowns
        correlation_change = False
        current_corrs = current_regime.get("correlations", {})
        previous_corrs = previous_regime.get("correlations", {})
        
        common_symbols = set(current_corrs.keys()).intersection(previous_corrs.keys())
        if common_symbols:
            avg_corr_change = np.mean([abs(current_corrs[s] - previous_corrs[s]) for s in common_symbols])
            correlation_change = avg_corr_change > 0.3  # Significant correlation change
        
        # Determine change type and severity
        change_detected = regime_changed or volatility_change or correlation_change
        
        change_type = []
        if regime_changed:
            change_type.append(f"regime_change_{previous_regime.get('regime', 'unknown')}_to_{current_regime.get('regime', 'unknown')}")
        
        if volatility_change:
            direction = "increased" if current_vol > previous_vol else "decreased"
            change_type.append(f"volatility_{direction}")
        
        if correlation_change:
            change_type.append("correlation_breakdown")
        
        # Calculate severity of change
        severity = 0.0
        if regime_changed:
            severity += 0.6
        
        if volatility_change:
            severity += min(0.3, abs(vol_change_pct))
        
        if correlation_change:
            severity += min(0.3, avg_corr_change if 'avg_corr_change' in locals() else 0)
        
        severity = min(1.0, severity)
        
        return {
            "change_detected": change_detected,
            "change_type": change_type,
            "severity": float(severity),
            "from_regime": previous_regime.get("regime"),
            "to_regime": current_regime.get("regime"),
            "volatility_change": float(vol_change_pct) if 'vol_change_pct' in locals() else 0,
            "timestamp": datetime.now().isoformat()
        }
    
    def calculate_market_strength_index(self, market_data: Dict[str, pd.DataFrame], 
                                       window: int = 14) -> Dict:
        """
        Calculates a composite market strength index based on multiple assets
        
        Args:
            market_data: Dictionary of DataFrames with OHLCV data
            window: Window period for calculations
            
        Returns:
            Dictionary with market strength analysis
        """
        result = {
            "strength_index": 50.0,  # Neutral by default
            "participating_assets": 0,
            "breadth": 0.0,
            "momentum": 0.0,
            "timestamp": datetime.now().isoformat()
        }
        
        try:
            if not market_data:
                result["error"] = "No market data provided"
                return result
            
            # 1. Calculate individual asset metrics
            asset_metrics = {}
            valid_assets = 0
            
            for symbol, df in market_data.items():
                if len(df) < window * 2:
                    continue
                
                valid_assets += 1
                metrics = {}
                
                # Price performance
                current_price = df['close'].iloc[-1]
                prev_price = df['close'].iloc[-window]
                performance = (current_price - prev_price) / prev_price * 100
                metrics["performance"] = performance
                
                # Above/below moving averages
                metrics["above_50ma"] = df['close'].iloc[-1] > talib.SMA(df['close'].values, timeperiod=min(50, len(df))).iloc[-1]
                metrics["above_200ma"] = df['close'].iloc[-1] > talib.SMA(df['close'].values, timeperiod=min(200, len(df))).iloc[-1]
                
                # RSI
                metrics["rsi"] = talib.RSI(df['close'].values, timeperiod=14).iloc[-1]
                
                # MACD
                macd, signal, hist = talib.MACD(
                    df['close'].values, 
                    fastperiod=12, 
                    slowperiod=26, 
                    signalperiod=9
                )
                metrics["macd_hist"] = hist[-1]
                metrics["macd_sign"] = 1 if hist[-1] > 0 else -1
                
                # Rate of change
                metrics["roc"] = talib.ROC(df['close'].values, timeperiod=10).iloc[-1]
                
                # Store metrics for this asset
                asset_metrics[symbol] = metrics
            
            # 2. Calculate aggregated metrics
            if not asset_metrics:
                result["error"] = "No valid assets with sufficient data"
                return result
            
            # Count assets above key moving averages
            above_50ma_count = sum(1 for metrics in asset_metrics.values() if metrics.get("above_50ma", False))
            above_200ma_count = sum(1 for metrics in asset_metrics.values() if metrics.get("above_200ma", False))
            
            # Calculate breadth (% of assets above moving averages)
            breadth_50ma = above_50ma_count / valid_assets * 100
            breadth_200ma = above_200ma_count / valid_assets * 100
            
            # Calculate average momentum indicators
            avg_rsi = np.mean([metrics.get("rsi", 50) for metrics in asset_metrics.values()])
            avg_roc = np.mean([metrics.get("roc", 0) for metrics in asset_metrics.values()])
            
            # Bullish vs bearish MACD count
            bullish_macd_count = sum(1 for metrics in asset_metrics.values() if metrics.get("macd_sign", 0) > 0)
            bearish_macd_count = valid_assets - bullish_macd_count
            
            # 3. Compute the market strength index (0-100 scale)
            # Component 1: Breadth (weight = 0.3)
            breadth_component = (breadth_50ma + breadth_200ma) / 2
            
            # Component 2: Momentum (weight = 0.4)
            momentum_component = (
                (avg_rsi - 30) / 40 * 100  # Scale RSI to 0-100
            )
            momentum_component = max(0, min(100, momentum_component))  # Clamp to 0-100
            
            # Component 3: Trend (weight = 0.3)
            trend_component = (bullish_macd_count / valid_assets * 100)
            
            # Combined strength index
            strength_index = (
                0.3 * breadth_component +
                0.4 * momentum_component +
                0.3 * trend_component
            )
            
            # 4. Prepare result
            result["strength_index"] = float(strength_index)
            result["participating_assets"] = valid_assets
            result["breadth"] = {
                "above_50ma_pct": float(breadth_50ma),
                "above_200ma_pct": float(breadth_200ma)
            }
            result["momentum"] = {
                "avg_rsi": float(avg_rsi),
                "avg_roc": float(avg_roc),
                "bullish_macd_pct": float(bullish_macd_count / valid_assets * 100)
            }
            result["interpretation"] = self._interpret_strength_index(strength_index)
            
        except Exception as e:
            logger.error(f"Error in market strength calculation: {str(e)}")
            result["error"] = f"Error in calculation: {str(e)}"
        
        return result
    
    def _interpret_strength_index(self, strength_index: float) -> Dict:
        """
        Interprets the market strength index value
        
        Args:
            strength_index: The calculated strength index (0-100)
            
        Returns:
            Dictionary with interpretation
        """
        if strength_index >= 80:
            return {
                "state": "strongly_bullish",
                "description": "Market showing strong bullish momentum across most assets",
                "action": "Consider trend-following strategies and long positions"
            }
        elif strength_index >= 60:
            return {
                "state": "bullish",
                "description": "Market showing positive momentum with broad participation",
                "action": "Look for pullbacks as buying opportunities"
            }
        elif strength_index >= 40:
            return {
                "state": "neutral",
                "description": "Market showing mixed signals with no clear direction",
                "action": "Exercise caution and focus on range-bound strategies"
            }
        elif strength_index >= 20:
            return {
                "state": "bearish",
                "description": "Market showing negative momentum across many assets",
                "action": "Consider defensive positions or selective short opportunities"
            }
        else:
            return {
                "state": "strongly_bearish",
                "description": "Market showing strong bearish momentum across most assets",
                "action": "Focus on capital preservation and potential short opportunities"
            }
    
    def calculate_cross_pair_correlations(self, market_data: Dict[str, pd.DataFrame], 
                                         period: int = 30) -> Dict:
        """
        Calculates correlations between different trading pairs
        
        Args:
            market_data: Dictionary of DataFrames with OHLCV data
            period: Period for correlation calculations
            
        Returns:
            Dictionary with correlation analysis
        """
        result = {
            "correlation_matrix": {},
            "highly_correlated_pairs": [],
            "inversely_correlated_pairs": [],
            "uncorrelated_pairs": [],
            "timestamp": datetime.now().isoformat()
        }
        
        try:
            if not market_data or len(market_data) < 2:
                result["error"] = "Need at least 2 assets for correlation analysis"
                return result
            
            # Extract returns for each asset
            returns_data = {}
            symbols = []
            
            for symbol, df in market_data.items():
                if len(df) < period:
                    continue
                    
                returns = df['close'].pct_change().dropna().iloc[-period:]
                if len(returns) == period:
                    returns_data[symbol] = returns
                    symbols.append(symbol)
            
            # Calculate correlation matrix
            correlation_matrix = {}
            
            for i, symbol1 in enumerate(symbols):
                correlation_matrix[symbol1] = {}
                
                for j, symbol2 in enumerate(symbols):
                    # Skip same symbol
                    if i == j:
                        correlation_matrix[symbol1][symbol2] = 1.0
                        continue
                    
                    # Calculate correlation
                    returns1 = returns_data[symbol1]
                    returns2 = returns_data[symbol2]
                    
                    # Ensure we're using the same timeframe
                    common_index = returns1.index.intersection(returns2.index)
                    if len(common_index) < period * 0.8:  # Require at least 80% of periods
                        correlation_matrix[symbol1][symbol2] = None
                        continue
                        
                    correlation = returns1.loc[common_index].corr(returns2.loc[common_index])
                    correlation_matrix[symbol1][symbol2] = float(correlation)
                    
                    # Categorize correlation
                    pair = (symbol1, symbol2)
                    
                    if abs(correlation) > 0.8:
                        if correlation > 0:
                            result["highly_correlated_pairs"].append({
                                "pair": pair,
                                "correlation": float(correlation)
                            })
                        else:
                            result["inversely_correlated_pairs"].append({
                                "pair": pair,
                                "correlation": float(correlation)
                            })
                    elif abs(correlation) < 0.2:
                        result["uncorrelated_pairs"].append({
                            "pair": pair,
                            "correlation": float(correlation)
                        })
            
            # Set correlation matrix and update count stats
            result["correlation_matrix"] = correlation_matrix
            result["high_correlation_count"] = len(result["highly_correlated_pairs"])
            result["inverse_correlation_count"] = len(result["inversely_correlated_pairs"])
            result["uncorrelated_count"] = len(result["uncorrelated_pairs"])
            
        except Exception as e:
            logger.error(f"Error in correlation calculation: {str(e)}")
            result["error"] = f"Error in calculation: {str(e)}"
        
        return result

    def detect_volume_anomalies(self, market_data: Dict[str, pd.DataFrame], 
                              lookback: int = 30, 
                              threshold: float = 3.0) -> Dict:
        """
        Detects unusual volume activity across multiple markets
        
        Args:
            market_data: Dictionary of DataFrames with OHLCV data
            lookback: Period for baseline volume calculation
            threshold: Multiplier threshold for anomaly detection
            
        Returns:
            Dictionary with volume anomaly analysis
        """
        result = {
            "anomalies_detected": False,
            "anomalies_count": 0,
            "assets_with_anomalies": [],
            "market_volume_surge": False,
            "timestamp": datetime.now().isoformat()
        }
        
        try:
            if not market_data:
                result["error"] = "No market data provided"
                return result
                
            # Analyze volume for each asset
            anomalies = []
            
            for symbol, df in market_data.items():
                if len(df) < lookback + 5:
                    continue
                
                # Calculate average volume over the lookback period
                avg_volume = df['volume'].iloc[-lookback-5:-5].mean()
                current_volume = df['volume'].iloc[-1]
                
                # Calculate relative volume
                if avg_volume > 0:
                    relative_volume = current_volume / avg_volume
                    
                    # Check for anomaly
                    if relative_volume > threshold:
                        # Check if price moved significantly with the volume
                        price_change = df['close'].iloc[-1] / df['close'].iloc[-2] - 1
                        
                        anomalies.append({
                            "symbol": symbol,
                            "relative_volume": float(relative_volume),
                            "price_change_pct": float(price_change * 100),
                            "direction": "bullish" if price_change > 0 else "bearish"
                        })
            
            # Calculate market-wide volume statistics
            if anomalies:
                result["anomalies_detected"] = True
                result["anomalies_count"] = len(anomalies)
                result["assets_with_anomalies"] = anomalies
                
                # Check if we have a market-wide volume surge
                # (defined as >20% of assets showing volume anomalies)
                if len(anomalies) / len(market_data) > 0.2:
                    result["market_volume_surge"] = True
                    
                    # Check if there's a clear direction to the surge
                    bullish_count = sum(1 for a in anomalies if a["direction"] == "bullish")
                    bearish_count = len(anomalies) - bullish_count
                    
                    if bullish_count > bearish_count * 2:
                        result["surge_direction"] = "bullish"
                    elif bearish_count > bullish_count * 2:
                        result["surge_direction"] = "bearish"
                    else:
                        result["surge_direction"] = "mixed"
        
        except Exception as e:
            logger.error(f"Error in volume anomaly detection: {str(e)}")
            result["error"] = f"Error in detection: {str(e)}"
        
        return result
