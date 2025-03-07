"""
Module for analyzing cross-market signals and intermarket relationships
to identify trading opportunities and market conditions
"""
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Union
from datetime import datetime, timedelta
import talib
from scipy import stats

from utils.logger import setup_logger
from utils.technical_analysis import TechnicalAnalysis
from utils.market_indicators import MarketIndicators

logger = setup_logger("cross_market_signals")

class CrossMarketSignals:
    """
    Class for analyzing relationships between different markets and generating
    cross-market trading signals
    """
    def __init__(self):
        """Initialize the cross-market signals analyzer"""
        self.market_indicators = MarketIndicators()
        self.correlation_threshold = 0.7
        self.lookback_periods = {
            "short": 14,
            "medium": 30,
            "long": 90
        }
        
    def analyze_intermarket_relationships(self, market_data: Dict[str, pd.DataFrame],
                                        target_symbol: str) -> Dict:
        """
        Analyzes relationships between markets to identify influences on target symbol
        
        Args:
            market_data: Dictionary of DataFrames with OHLCV data for different markets
            target_symbol: Symbol to analyze influences for
            
        Returns:
            Dictionary with intermarket relationship analysis
        """
        result = {
            "target_symbol": target_symbol,
            "leading_indicators": [],
            "correlated_markets": [],
            "inversely_correlated": [],
            "dominant_influence": None,
            "timestamp": datetime.now().isoformat()
        }
        
        try:
            if target_symbol not in market_data or market_data[target_symbol].empty:
                result["error"] = f"Target symbol {target_symbol} not found in market data"
                return result
            
            # Get target price data
            target_data = market_data[target_symbol]
            target_returns = target_data['close'].pct_change().dropna()
            
            # Initialize structures for analysis
            correlations = {}
            lead_lag = {}
            
            # Analyze relationship with each other market
            for symbol, df in market_data.items():
                if symbol == target_symbol or df.empty:
                    continue
                
                # Calculate returns
                symbol_returns = df['close'].pct_change().dropna()
                
                # Find common time index
                common_index = target_returns.index.intersection(symbol_returns.index)
                if len(common_index) < 30:  # Need minimum data points
                    continue
                
                # Calculate correlations for different periods
                corr_data = {}
                for period_name, period in self.lookback_periods.items():
                    if len(common_index) < period:
                        continue
                        
                    # Get the latest period data
                    period_idx = common_index[-period:]
                    
                    # Calculate correlation
                    correlation = target_returns.loc[period_idx].corr(symbol_returns.loc[period_idx])
                    corr_data[period_name] = float(correlation)
                
                # Store correlation data
                correlations[symbol] = corr_data
                
                # Calculate lead-lag relationship (which one leads the other)
                max_lag = 10  # Maximum lag to check (in periods)
                xcorr = {}
                
                for lag in range(-max_lag, max_lag + 1):
                    if lag == 0:
                        # Zero lag is just the standard correlation
                        xcorr[lag] = corr_data.get("short", 0)
                        continue
                    
                    # Calculate cross-correlation
                    if lag > 0:
                        # symbol_returns leads target_returns
                        lagged_target = target_returns.loc[common_index].shift(-lag)
                        correlation = symbol_returns.loc[common_index].corr(lagged_target.dropna())
                    else:
                        # target_returns leads symbol_returns
                        lagged_symbol = symbol_returns.loc[common_index].shift(-lag)
                        correlation = target_returns.loc[common_index].corr(lagged_symbol.dropna())
                    
                    xcorr[lag] = float(correlation) if not np.isnan(correlation) else 0
                
                # Find the lag with the maximum correlation
                max_corr_lag = max(xcorr.items(), key=lambda x: abs(x[1]))
                
                lead_lag[symbol] = {
                    "max_correlation_lag": max_corr_lag[0],
                    "max_correlation_value": max_corr_lag[1],
                    "leads_target": max_corr_lag[0] > 0 and abs(max_corr_lag[1]) > self.correlation_threshold,
                    "led_by_target": max_corr_lag[0] < 0 and abs(max_corr_lag[1]) > self.correlation_threshold
                }
            
            # 3. Identify leading indicators (assets that lead the target)
            for symbol, lag_data in lead_lag.items():
                if lag_data["leads_target"]:
                    correlation = lag_data["max_correlation_value"]
                    relationship = "positive" if correlation > 0 else "negative"
                    
                    result["leading_indicators"].append({
                        "symbol": symbol,
                        "lag": lag_data["max_correlation_lag"],
                        "correlation": correlation,
                        "relationship": relationship,
                        "description": f"{symbol} leads {target_symbol} by {lag_data['max_correlation_lag']} periods ({relationship} correlation)"
                    })
            
            # 4. Identify correlated and inversely correlated markets
            for symbol, corr_data in correlations.items():
                medium_corr = corr_data.get("medium", 0)
                
                if abs(medium_corr) > self.correlation_threshold:
                    market_info = {
                        "symbol": symbol,
                        "correlation": medium_corr,
                        "short_term_correlation": corr_data.get("short", 0),
                        "long_term_correlation": corr_data.get("long", 0),
                        "correlation_stability": self._calculate_correlation_stability(
                            corr_data.get("short", 0),
                            corr_data.get("medium", 0),
                            corr_data.get("long", 0)
                        )
                    }
                    
                    if medium_corr > 0:
                        result["correlated_markets"].append(market_info)
                    else:
                        result["inversely_correlated"].append(market_info)
            
            # 5. Determine the dominant influence
            if result["leading_indicators"]:
                # Sort by absolute correlation value
                sorted_leaders = sorted(
                    result["leading_indicators"], 
                    key=lambda x: abs(x["correlation"]), 
                    reverse=True
                )
                result["dominant_influence"] = sorted_leaders[0]
            
            # 6. Sort correlation lists by strength
            result["correlated_markets"] = sorted(
                result["correlated_markets"],
                key=lambda x: abs(x["correlation"]),
                reverse=True
            )
            
            result["inversely_correlated"] = sorted(
                result["inversely_correlated"],
                key=lambda x: abs(x["correlation"]),
                reverse=True
            )
            
        except Exception as e:
            logger.error(f"Error in intermarket relationship analysis: {str(e)}")
            result["error"] = f"Error in analysis: {str(e)}"
        
        return result
    
    def _calculate_correlation_stability(self, short_corr: float, medium_corr: float, long_corr: float) -> float:
        """
        Calculates how stable the correlation is across different timeframes
        
        Args:
            short_corr: Short term correlation
            medium_corr: Medium term correlation
            long_corr: Long term correlation
            
        Returns:
            Stability score (0-1 where 1 is most stable)
        """
        if short_corr is None or medium_corr is None or long_corr is None:
            return 0.0
            
        # Calculate standard deviation of correlations
        correlations = [c for c in [short_corr, medium_corr, long_corr] if c is not None]
        if not correlations:
            return 0.0
            
        std_dev = np.std(correlations)
        
        # Transform to stability score (1 - normalized std_dev)
        # Max std_dev would be 2 (if correlations range from -1 to 1)
        stability = 1.0 - min(1.0, std_dev / 2.0)
        
        return float(stability)
    
    def generate_cross_market_signals(self, market_data: Dict[str, pd.DataFrame],
                                    target_symbol: str) -> Dict:
        """
        Generates trading signals based on cross-market relationships
        
        Args:
            market_data: Dictionary of DataFrames with OHLCV data for different markets
            target_symbol: Symbol to generate signals for
            
        Returns:
            Dictionary with trading signals and analysis
        """
        result = {
            "target_symbol": target_symbol,
            "signals": [],
            "signal_strength": 0,
            "signal_direction": "neutral",
            "confidence": 0.0,
            "timestamp": datetime.now().isoformat()
        }
        
        try:
            # 1. Analyze intermarket relationships
            relationships = self.analyze_intermarket_relationships(market_data, target_symbol)
            
            if "error" in relationships:
                result["error"] = relationships["error"]
                return result
            
            # 2. Get leading indicators
            leading_indicators = relationships.get("leading_indicators", [])
            
            # If no leading indicators, can't generate signals
            if not leading_indicators:
                result["message"] = "No leading indicators found for signal generation"
                return result
            
            # 3. Analyze leading indicators for signals
            signal_count = 0
            bullish_signals = 0
            bearish_signals = 0
            
            for leader in leading_indicators:
                symbol = leader["symbol"]
                lag = leader["lag"]
                relationship = leader["relationship"]
                
                if symbol not in market_data or lag <= 0:
                    continue
                
                # Get data for leading market
                leader_data = market_data[symbol]
                
                # Calculate short-term momentum
                leader_close = leader_data['close'].values
                leader_rsi = talib.RSI(leader_close, timeperiod=14)[-1]
                
                # Short-term trend direction
                leader_sma_fast = talib.SMA(leader_close, timeperiod=5)[-1]
                leader_sma_slow = talib.SMA(leader_close, timeperiod=20)[-1]
                
                trend_direction = "up" if leader_sma_fast > leader_sma_slow else "down"
                
                # Determine signal based on relationship type and trend direction
                signal_direction = ""
                if relationship == "positive":
                    signal_direction = trend_direction
                else:  # negative relationship
                    signal_direction = "up" if trend_direction == "down" else "down"
                
                # Calculate signal strength based on RSI and trend strength
                trend_strength = abs((leader_sma_fast / leader_sma_slow - 1) * 100)
                rsi_strength = abs(leader_rsi - 50) / 50
                
                signal_strength = (trend_strength * 0.6 + rsi_strength * 0.4) / 100
                
                # Adjust for correlation strength
                correlation_factor = abs(leader["correlation"])
                signal_strength *= correlation_factor
                
                # Convert to -1 to +1 scale
                if signal_direction == "down":
                    signal_strength *= -1
                
                # Add to signals
                result["signals"].append({
                    "from_symbol": symbol,
                    "lag_periods": lag,
                    "relationship": relationship,
                    "direction": signal_direction,
                    "strength": float(signal_strength),
                    "correlation": float(leader["correlation"]),
                    "importance": float(correlation_factor)
                })
                
                # Track signal counts
                signal_count += 1
                if signal_strength > 0:
                    bullish_signals += signal_strength
                else:
                    bearish_signals -= signal_strength
            
            # 4. Determine overall signal
            if signal_count > 0:
                # Net signal strength (-1 to +1 scale)
                net_signal = (bullish_signals - bearish_signals) / signal_count
                
                # Determine direction and confidence
                result["signal_strength"] = float(net_signal)
                result["signal_direction"] = "bullish" if net_signal > 0 else "bearish" if net_signal < 0 else "neutral"
                result["confidence"] = min(1.0, abs(net_signal) * 2)  # Scale to 0-1
                
                # Add overall interpretation
                if abs(net_signal) > 0.5:
                    strength_desc = "strong"
                elif abs(net_signal) > 0.2:
                    strength_desc = "moderate"
                else:
                    strength_desc = "weak"
                
                result["interpretation"] = f"{strength_desc} {result['signal_direction']} signal from leading indicators"
            else:
                result["interpretation"] = "No actionable cross-market signals detected"
            
            # 5. Add the relationships for reference
            result["relationships"] = {
                "leading_indicators": relationships["leading_indicators"],
                "dominant_influence": relationships["dominant_influence"]
            }
            
        except Exception as e:
            logger.error(f"Error in cross-market signal generation: {str(e)}")
            result["error"] = f"Error in signal generation: {str(e)}"
        
        return result
    
    def detect_market_rotation(self, market_data: Dict[str, pd.DataFrame],
                             lookback_period: int = 30,
                             short_term_period: int = 5) -> Dict:
        """
        Detects rotation between different market segments/assets
        
        Args:
            market_data: Dictionary of DataFrames with OHLCV data for different markets
            lookback_period: Period for baseline performance
            short_term_period: Period for recent performance comparison
            
        Returns:
            Dictionary with market rotation analysis
        """
        result = {
            "rotation_detected": False,
            "from_sectors": [],
            "to_sectors": [],
            "rotation_strength": 0.0,
            "rotations": [],
            "timestamp": datetime.now().isoformat()
        }
        
        try:
            if not market_data or len(market_data) < 4:  # Need at least a few markets to detect rotation
                result["error"] = "Insufficient market data for rotation detection"
                return result
            
            # 1. Calculate performance metrics for each market/asset
            performance = {}
            
            for symbol, df in market_data.items():
                if len(df) < lookback_period + short_term_period:
                    continue
                
                # Calculate baseline performance
                baseline_start = df['close'].iloc[-lookback_period-short_term_period]
                baseline_end = df['close'].iloc[-short_term_period]
                baseline_perf = (baseline_end / baseline_start - 1) * 100
                
                # Calculate recent performance
                recent_start = df['close'].iloc[-short_term_period]
                recent_end = df['close'].iloc[-1]
                recent_perf = (recent_end / recent_start - 1) * 100
                
                # Momentum shift
                momentum_shift = recent_perf - baseline_perf
                
                # Store metrics
                performance[symbol] = {
                    "baseline_performance": float(baseline_perf),
                    "recent_performance": float(recent_perf),
                    "momentum_shift": float(momentum_shift),
                    "volume_change": float(df['volume'].iloc[-short_term_period:].mean() / 
                                           df['volume'].iloc[-lookback_period-short_term_period:-short_term_period].mean() - 1) * 100
                }
            
            # 2. Identify top gainers and losers based on momentum shift
            sorted_shifts = sorted(
                [(symbol, data["momentum_shift"]) for symbol, data in performance.items()],
                key=lambda x: x[1],
                reverse=True
            )
            
            # Get top and bottom quartiles
            top_quartile = sorted_shifts[:max(1, len(sorted_shifts)//4)]
            bottom_quartile = sorted_shifts[-max(1, len(sorted_shifts)//4):]
            
            # 3. Detect if there's significant rotation
            if top_quartile and bottom_quartile:
                avg_top_shift = np.mean([shift for _, shift in top_quartile])
                avg_bottom_shift = np.mean([shift for _, shift in bottom_quartile])
                
                # Rotation strength
                rotation_strength = avg_top_shift - avg_bottom_shift
                
                # Consider it a rotation if the difference is significant
                result["rotation_detected"] = rotation_strength > 10  # 10% difference threshold
                result["rotation_strength"] = float(rotation_strength)
                
                # Identify the rotation direction
                if result["rotation_detected"]:
                    result["from_sectors"] = [symbol for symbol, _ in bottom_quartile]
                    result["to_sectors"] = [symbol for symbol, _ in top_quartile]
                    
                    # Document individual rotations
                    for to_symbol, to_shift in top_quartile:
                        for from_symbol, from_shift in bottom_quartile:
                            result["rotations"].append({
                                "from": from_symbol,
                                "to": to_symbol,
                                "magnitude": float(to_shift - from_shift),
                                "from_performance": performance[from_symbol],
                                "to_performance": performance[to_symbol]
                            })
                    
                    # Sort rotations by magnitude
                    result["rotations"].sort(key=lambda x: x["magnitude"], reverse=True)
                    
            # 4. Add performance data for reference
            result["all_performance"] = performance
            
        except Exception as e:
            logger.error(f"Error in market rotation detection: {str(e)}")
            result["error"] = f"Error in rotation detection: {str(e)}"
        
        return result
    
    def analyze_cross_market_divergence(self, market_data: Dict[str, pd.DataFrame],
                                      reference_pairs: List[Tuple[str, str]] = None) -> Dict:
        """
        Analyzes divergences between typically correlated markets
        
        Args:
            market_data: Dictionary of DataFrames with OHLCV data for different markets
            reference_pairs: List of symbol pairs that are typically correlated
            
        Returns:
            Dictionary with cross-market divergence analysis
        """
        result = {
            "divergences": [],
            "convergences": [],
            "correlations": {},
            "timestamp": datetime.now().isoformat()
        }
        
        try:
            if not market_data:
                result["error"] = "No market data provided"
                return result
                
            # If no reference pairs provided, use all possible combinations
            if reference_pairs is None:
                symbols = list(market_data.keys())
                reference_pairs = []
                
                for i in range(len(symbols)):
                    for j in range(i+1, len(symbols)):
                        reference_pairs.append((symbols[i], symbols[j]))
            
            # Calculate current and historical correlations for each pair
            for pair in reference_pairs:
                symbol1, symbol2 = pair
                
                if symbol1 not in market_data or symbol2 not in market_data:
                    continue
                
                df1 = market_data[symbol1]
                df2 = market_data[symbol2]
                
                # Ensure sufficient data
                min_length = 60  # Need at least this many periods
                if len(df1) < min_length or len(df2) < min_length:
                    continue
                
                # Calculate returns
                returns1 = df1['close'].pct_change().dropna()
                returns2 = df2['close'].pct_change().dropna()
                
                # Find common time index
                common_index = returns1.index.intersection(returns2.index)
                if len(common_index) < min_length:
                    continue
                
                # Get returns aligned on common_index
                aligned_returns1 = returns1.loc[common_index]
                aligned_returns2 = returns2.loc[common_index]
                
                # Calculate correlations for different periods
                short_period = 14
                medium_period = 30
                long_period = 90
                
                # Long-term "normal" correlation
                if len(common_index) >= long_period:
                    long_term_corr = aligned_returns1.iloc[-long_period:].corr(aligned_returns2.iloc[-long_period:])
                else:
                    long_term_corr = aligned_returns1.corr(aligned_returns2)
                
                # Medium-term correlation
                if len(common_index) >= medium_period:
                    medium_term_corr = aligned_returns1.iloc[-medium_period:].corr(aligned_returns2.iloc[-medium_period:])
                else:
                    medium_term_corr = long_term_corr
                
                # Short-term recent correlation
                if len(common_index) >= short_period:
                    short_term_corr = aligned_returns1.iloc[-short_period:].corr(aligned_returns2.iloc[-short_period:])
                else:
                    short_term_corr = medium_term_corr
                
                # Store correlation data
                correlation_data = {
                    "pair": pair,
                    "long_term": float(long_term_corr),
                    "medium_term": float(medium_term_corr),
                    "short_term": float(short_term_corr),
                    "current_correlation": float(short_term_corr)
                }
                
                pair_key = f"{symbol1}_{symbol2}"
                result["correlations"][pair_key] = correlation_data
                
                # Check for divergences (shift in correlation)
                correlation_shift = short_term_corr - long_term_corr
                
                # Consider it a divergence if the shift is significant
                if abs(correlation_shift) > 0.4:  # Significant shift in correlation
                    divergence = {
                        "pair": pair,
                        "correlation_shift": float(correlation_shift),
                        "long_term_correlation": float(long_term_corr),
                        "current_correlation": float(short_term_corr),
                        "severity": "high" if abs(correlation_shift) > 0.7 else "medium",
                        "type": "breakdown" if correlation_shift < 0 else "strengthening"
                    }
                    
                    # Check price action divergence
                    price_change1 = (df1['close'].iloc[-1] / df1['close'].iloc[-short_period] - 1) * 100
                    price_change2 = (df2['close'].iloc[-1] / df2['close'].iloc[-short_period] - 1) * 100
                    
                    # If prices are moving in opposite directions
                    if price_change1 * price_change2 < 0:
                        divergence["price_divergence"] = True
                        divergence["price_change"] = {
                            symbol1: float(price_change1),
                            symbol2: float(price_change2)
                        }
                    else:
                        divergence["price_divergence"] = False
                    
                    result["divergences"].append(divergence)
                
                # Check for convergence (returning to historical correlation)
                elif abs(medium_term_corr - long_term_corr) > 0.4 and abs(short_term_corr - long_term_corr) < 0.2:
                    result["convergences"].append({
                        "pair": pair,
                        "previous_divergence": float(medium_term_corr - long_term_corr),
                        "current_alignment": float(short_term_corr - long_term_corr),
                        "long_term_correlation": float(long_term_corr),
                        "current_correlation": float(short_term_corr)
                    })
            
            # Sort divergences by severity
            result["divergences"].sort(key=lambda x: abs(x["correlation_shift"]), reverse=True)
            
            # Calculate market implications
            if result["divergences"]:
                # Check if there are multiple severe divergences (market regime shift)
                severe_count = sum(1 for d in result["divergences"] if d["severity"] == "high")
                
                if severe_count >= 2:
                    result["market_implication"] = "Potential market regime change with multiple correlation breakdowns"
                else:
                    result["market_implication"] = "Isolated correlation shifts, monitor for broader impact"
            
        except Exception as e:
            logger.error(f"Error in cross-market divergence analysis: {str(e)}")
            result["error"] = f"Error in divergence analysis: {str(e)}"
        
        return result