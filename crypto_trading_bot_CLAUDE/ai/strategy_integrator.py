"""
Strategy integrator to combine AI predictions with traditional technical strategies
for more robust trading signals
"""
import os
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Union, Optional, Any
from datetime import datetime

from config.config import DATA_DIR
from utils.logger import setup_logger
from ai.prediction_orchestrator import PredictionOrchestrator
from strategies.technical_bounce import detect_technical_bounce
from strategies.market_state import analyze_market_state
from strategies.support_resistance import identify_support_resistance
from strategies.volume_profile import analyze_volume_profile
from ai.scoring_engine import ScoringEngine

logger = setup_logger("strategy_integrator")

class StrategyIntegrator:
    """
    Integrates AI predictions with traditional technical analysis strategies
    to generate more robust trading signals. Acts as middleware between 
    prediction orchestrator and trading agent.
    """
    def __init__(self, 
                prediction_orchestrator: Optional[PredictionOrchestrator] = None,
                scoring_engine: Optional[ScoringEngine] = None):
        """
        Initialize the strategy integrator
        
        Args:
            prediction_orchestrator: Prediction orchestrator instance
            scoring_engine: Scoring engine instance
        """
        self.prediction_orchestrator = prediction_orchestrator or PredictionOrchestrator()
        self.scoring_engine = scoring_engine or ScoringEngine()
        
        # Strategy weights for ensemble
        self.strategy_weights = {
            "ai_prediction": 0.60,
            "technical_bounce": 0.15,
            "support_resistance": 0.15,
            "volume_profile": 0.10
        }
        
        # Performance tracking for weight adjustment
        self.strategy_performance = {
            "ai_prediction": {"correct": 0, "incorrect": 0, "total_pnl": 0.0},
            "technical_bounce": {"correct": 0, "incorrect": 0, "total_pnl": 0.0},
            "support_resistance": {"correct": 0, "incorrect": 0, "total_pnl": 0.0},
            "volume_profile": {"correct": 0, "incorrect": 0, "total_pnl": 0.0}
        }
        
        # Threshold configuration
        self.confidence_thresholds = {
            "strong_buy": 0.75,
            "buy": 0.6,
            "neutral": 0.4,
            "sell": 0.6,
            "strong_sell": 0.75
        }
        
        # Strategy history for analysis
        self.signal_history = []
        self.max_history_size = 1000
        
        # Output directory
        self.output_dir = os.path.join(DATA_DIR, "strategy_signals")
        os.makedirs(self.output_dir, exist_ok=True)
    
    def generate_trade_signal(self, symbol: str, 
                           data: pd.DataFrame, 
                           timeframe: str = '1h',
                           additional_context: Dict = None) -> Dict:
        """
        Generate a unified trade signal by integrating AI predictions with
        traditional technical strategies
        
        Args:
            symbol: Trading symbol (e.g. 'BTCUSDT')
            data: DataFrame with OHLCV data
            timeframe: Data timeframe
            additional_context: Additional market context
            
        Returns:
            Dictionary with the integrated trade signal
        """
        # 1. Get AI prediction
        ai_prediction = self._get_ai_prediction(symbol, data, timeframe)
        
        # 2. Get technical analysis signals
        tech_signals = self._get_technical_signals(symbol, data, timeframe)
        
        # 3. Integrate signals using weighted ensemble
        integrated_signal = self._integrate_signals(
            ai_prediction, 
            tech_signals,
            symbol,
            timeframe,
            additional_context
        )
        
        # 4. Add the signal to history
        self._add_to_history(symbol, integrated_signal, ai_prediction, tech_signals)
        
        return integrated_signal
    
    def _get_ai_prediction(self, symbol: str, data: pd.DataFrame, timeframe: str) -> Dict:
        """
        Get prediction from the AI prediction orchestrator
        
        Args:
            symbol: Trading symbol
            data: OHLCV data
            timeframe: Data timeframe
            
        Returns:
            AI prediction result
        """
        try:
            prediction_result = self.prediction_orchestrator.get_prediction(
                symbol, data, timeframe, include_details=True
            )
            
            if prediction_result.get("success", False):
                return prediction_result
            else:
                logger.warning(f"Failed to get AI prediction for {symbol}: {prediction_result.get('error', 'Unknown error')}")
                return {"success": False, "error": prediction_result.get("error", "Unknown error")}
        except Exception as e:
            logger.error(f"Error getting AI prediction for {symbol}: {str(e)}")
            return {"success": False, "error": str(e)}
    
    def _get_technical_signals(self, symbol: str, data: pd.DataFrame, timeframe: str) -> Dict:
        """
        Get signals from traditional technical strategies
        
        Args:
            symbol: Trading symbol
            data: OHLCV data
            timeframe: Data timeframe
            
        Returns:
            Technical signals from various strategies
        """
        technical_signals = {}
        
        # 1. Market state analysis
        try:
            market_state = analyze_market_state(data)
            technical_signals["market_state"] = market_state
        except Exception as e:
            logger.error(f"Error analyzing market state for {symbol}: {str(e)}")
            technical_signals["market_state"] = {"state": "unknown", "error": str(e)}
        
        # 2. Technical bounce detection
        try:
            bounce_result = detect_technical_bounce(data, technical_signals["market_state"])
            technical_signals["technical_bounce"] = bounce_result
        except Exception as e:
            logger.error(f"Error detecting technical bounce for {symbol}: {str(e)}")
            technical_signals["technical_bounce"] = {"signals": [], "error": str(e)}
        
        # 3. Support/resistance identification
        try:
            sr_levels = identify_support_resistance(data)
            technical_signals["support_resistance"] = sr_levels
        except Exception as e:
            logger.error(f"Error identifying support/resistance for {symbol}: {str(e)}")
            technical_signals["support_resistance"] = {"levels": [], "error": str(e)}
        
        # 4. Volume profile analysis
        try:
            volume_profile = analyze_volume_profile(data)
            technical_signals["volume_profile"] = volume_profile
        except Exception as e:
            logger.error(f"Error analyzing volume profile for {symbol}: {str(e)}")
            technical_signals["volume_profile"] = {"profile": {}, "error": str(e)}
        
        # Calculate technical score based on these signals
        technical_signals["score"] = self._calculate_technical_score(technical_signals, data)
        
        return technical_signals
    
    def _calculate_technical_score(self, technical_signals: Dict, data: pd.DataFrame) -> Dict:
        """
        Calculate score from technical signals
        
        Args:
            technical_signals: Dictionary of technical signals
            data: OHLCV data
            
        Returns:
            Technical score dictionary
        """
        # Get current price
        current_price = float(data['close'].iloc[-1]) if not data.empty else None
        
        # Initialize scores
        bullish_score = 0.0
        bearish_score = 0.0
        total_weight = 0.0
        signals = []
        
        # 1. Market state contribution
        market_state = technical_signals.get("market_state", {})
        state = market_state.get("state", "unknown")
        
        # State weights - how much we trust signals in each market state
        state_weights = {
            "bullish": 1.0,
            "bearish": 1.0, 
            "sideways": 0.7,
            "volatile": 0.5,
            "unknown": 0.3
        }
        
        weight = state_weights.get(state, 0.5)
        trend_strength = market_state.get("trend_strength", 0.5)
        
        if state == "bullish":
            bullish_score += trend_strength * weight
            signals.append("bullish_trend")
        elif state == "bearish":
            bearish_score += trend_strength * weight
            signals.append("bearish_trend")
        
        total_weight += weight
        
        # 2. Technical bounce contribution
        bounce_result = technical_signals.get("technical_bounce", {})
        bounce_signals = bounce_result.get("signals", [])
        
        if "bullish_bounce" in bounce_signals:
            bullish_score += 0.7 * weight
            signals.append("bullish_bounce")
        if "bearish_bounce" in bounce_signals:
            bearish_score += 0.7 * weight
            signals.append("bearish_bounce")
        if "bullish_breakout" in bounce_signals:
            bullish_score += 0.8 * weight
            signals.append("bullish_breakout")
        if "bearish_breakdown" in bounce_signals:
            bearish_score += 0.8 * weight
            signals.append("bearish_breakdown")
        
        # 3. Support/resistance contribution
        sr_levels = technical_signals.get("support_resistance", {}).get("levels", [])
        
        if current_price is not None and sr_levels:
            # Find the closest levels
            levels_above = [l for l in sr_levels if l["price"] > current_price]
            levels_below = [l for l in sr_levels if l["price"] < current_price]
            
            if levels_above:
                closest_resistance = min(levels_above, key=lambda x: abs(x["price"] - current_price))
                resistance_distance = (closest_resistance["price"] - current_price) / current_price
                
                # If price is close to resistance, bearish signal
                if resistance_distance < 0.02:  # Within 2%
                    bearish_score += 0.6 * weight
                    signals.append("near_resistance")
            
            if levels_below:
                closest_support = max(levels_below, key=lambda x: x["price"])
                support_distance = (current_price - closest_support["price"]) / current_price
                
                # If price is close to support, bullish signal
                if support_distance < 0.02:  # Within 2%
                    bullish_score += 0.6 * weight
                    signals.append("near_support")
        
        # 4. Volume profile contribution
        volume_profile = technical_signals.get("volume_profile", {})
        poc_price = volume_profile.get("poc_price")  # Point of control
        
        if current_price is not None and poc_price is not None:
            # Distance to high volume node
            poc_distance = abs(current_price - poc_price) / current_price
            
            # If price is near a high volume node, might reverse to it
            if poc_distance < 0.03:  # Within 3%
                if current_price < poc_price:
                    bullish_score += 0.4 * weight
                    signals.append("below_volume_poc")
                else:
                    bearish_score += 0.4 * weight
                    signals.append("above_volume_poc")
        
        # Calculate final score (0-100 scale, 50 is neutral)
        if total_weight > 0:
            bullish_influence = bullish_score / total_weight
            bearish_influence = bearish_score / total_weight
            
            # Score: 0 = very bearish, 100 = very bullish
            final_score = 50 + (bullish_influence - bearish_influence) * 50
            final_score = max(0, min(100, final_score))
        else:
            final_score = 50  # Neutral if no signals
        
        # Determine direction based on score
        if final_score >= 70:
            direction = "BUY"
        elif final_score <= 30:
            direction = "SELL"
        else:
            direction = "NEUTRAL"
        
        return {
            "score": final_score,
            "direction": direction,
            "bullish_influence": bullish_score,
            "bearish_influence": bearish_score,
            "signals": signals,
            "strength": abs(final_score - 50) / 50  # 0-1 scale
        }
    
    def _integrate_signals(self, ai_prediction: Dict, tech_signals: Dict, 
                         symbol: str, timeframe: str, 
                         additional_context: Dict = None) -> Dict:
        """
        Integrate AI predictions with technical signals
        
        Args:
            ai_prediction: AI prediction result
            tech_signals: Technical signals result
            symbol: Trading symbol
            timeframe: Data timeframe
            additional_context: Additional market context
            
        Returns:
            Integrated trade signal
        """
        # Handle case where AI prediction failed
        ai_success = ai_prediction.get("success", False)
        
        if not ai_success:
            # If AI prediction failed, use only technical signals
            logger.warning(f"Using only technical signals for {symbol} due to AI prediction failure")
            return self._generate_signal_from_technical(tech_signals, symbol, timeframe)
        
        # Get AI prediction values
        prediction = ai_prediction.get("prediction", {})
        ai_direction = prediction.get("direction", "NEUTRAL")
        ai_confidence = prediction.get("confidence", 0.5)
        ai_strength = prediction.get("strength", 0)
        
        # Get technical score
        tech_score = tech_signals.get("score", {})
        tech_direction = tech_score.get("direction", "NEUTRAL")
        tech_strength = tech_score.get("strength", 0.0)
        
        # Get current weights
        ai_weight = self.strategy_weights.get("ai_prediction", 0.6)
        tech_bounce_weight = self.strategy_weights.get("technical_bounce", 0.15)
        sr_weight = self.strategy_weights.get("support_resistance", 0.15)
        vol_weight = self.strategy_weights.get("volume_profile", 0.10)
        
        # Calculate weighted direction score
        # 0 = strong sell, 0.5 = neutral, 1 = strong buy
        ai_score = 0.5  # Neutral
        if ai_direction == "BUY":
            ai_score = 0.5 + (ai_confidence - 0.5) * 2 * 0.5  # Map to 0.5-1.0
        elif ai_direction == "SELL":
            ai_score = 0.5 - (ai_confidence - 0.5) * 2 * 0.5  # Map to 0.0-0.5
        
        # Convert technical score to same scale (0-1)
        tech_norm_score = tech_score.get("score", 50) / 100
        
        # Weighted ensemble
        weighted_score = (
            ai_weight * ai_score +
            (tech_bounce_weight + sr_weight + vol_weight) * tech_norm_score
        )
        
        # Determine final direction and confidence
        # Map weighted_score back to direction and confidence
        if weighted_score > 0.5:
            # Bullish signal
            direction = "BUY"
            confidence = (weighted_score - 0.5) * 2  # Convert 0.5-1.0 to 0.0-1.0
            
            # Strong buy if above threshold
            if confidence >= self.confidence_thresholds["strong_buy"]:
                signal_type = "STRONG_BUY"
            else:
                signal_type = "BUY"
                
        elif weighted_score < 0.5:
            # Bearish signal
            direction = "SELL"
            confidence = (0.5 - weighted_score) * 2  # Convert 0.0-0.5 to 0.0-1.0
            
            # Strong sell if above threshold
            if confidence >= self.confidence_thresholds["strong_sell"]:
                signal_type = "STRONG_SELL"
            else:
                signal_type = "SELL"
                
        else:
            # Neutral
            direction = "NEUTRAL"
            confidence = 0.0
            signal_type = "NEUTRAL"
        
        # Calculate signal strength (0-10 scale)
        signal_strength = int(abs(weighted_score - 0.5) * 20)
        
        # Combine signals from all sources
        combined_signals = []
        if ai_success:
            combined_signals.extend(prediction.get("signals", []))
        combined_signals.extend(tech_score.get("signals", []))
        
        # Check if AI and technical are aligned
        signals_aligned = direction == ai_direction == tech_direction and direction != "NEUTRAL"
        
        # Adjust confidence if signals are aligned
        if signals_aligned:
            confidence = min(1.0, confidence * 1.2)  # Boost confidence by 20% if aligned
            combined_signals.append("ai_tech_aligned")
        
        # Create unified signal result
        signal = {
            "symbol": symbol,
            "timeframe": timeframe,
            "direction": direction,
            "signal_type": signal_type,
            "confidence": confidence,
            "strength": signal_strength,
            "signals": combined_signals,
            "timestamp": datetime.now().isoformat()
        }
        
        # Add context information
        if additional_context:
            signal["context"] = additional_context
        
        # Add contributing sources
        signal["contributors"] = {
            "ai_prediction": {
                "direction": ai_direction,
                "confidence": ai_confidence,
                "strength": ai_strength,
                "weight": ai_weight
            },
            "technical": {
                "direction": tech_direction,
                "score": tech_score.get("score", 50),
                "strength": tech_strength,
                "weight": tech_bounce_weight + sr_weight + vol_weight
            }
        }
        
        return signal
    
    def _generate_signal_from_technical(self, tech_signals: Dict, symbol: str, timeframe: str) -> Dict:
        """
        Generate a signal based only on technical analysis when AI is unavailable
        
        Args:
            tech_signals: Technical signals result
            symbol: Trading symbol
            timeframe: Data timeframe
            
        Returns:
            Technical-only trade signal
        """
        # Get technical score
        tech_score = tech_signals.get("score", {})
        score_value = tech_score.get("score", 50)
        direction = tech_score.get("direction", "NEUTRAL")
        
        # Calculate confidence based on distance from neutral
        confidence = abs(score_value - 50) / 50
        
        # Determine signal type
        if score_value >= 70:
            signal_type = "STRONG_BUY" if score_value >= 85 else "BUY"
        elif score_value <= 30:
            signal_type = "STRONG_SELL" if score_value <= 15 else "SELL"
        else:
            signal_type = "NEUTRAL"
        
        # Signal strength (0-10 scale)
        signal_strength = int(abs(score_value - 50) / 5)
        
        # Create signal
        signal = {
            "symbol": symbol,
            "timeframe": timeframe,
            "direction": direction,
            "signal_type": signal_type,
            "confidence": confidence,
            "strength": signal_strength,
            "signals": tech_score.get("signals", []),
            "timestamp": datetime.now().isoformat(),
            "technical_only": True
        }
        
        # Add contributing sources
        signal["contributors"] = {
            "ai_prediction": {
                "available": False
            },
            "technical": {
                "direction": direction,
                "score": score_value,
                "strength": tech_score.get("strength", 0.0),
                "weight": 1.0
            }
        }
        
        return signal
    
    def _add_to_history(self, symbol: str, integrated_signal: Dict, 
                      ai_prediction: Dict, tech_signals: Dict) -> None:
        """
        Add the signal to history for analysis and strategy improvement
        
        Args:
            symbol: Trading symbol
            integrated_signal: Integrated signal result
            ai_prediction: AI prediction result
            tech_signals: Technical signals result
        """
        # Create history entry with only the necessary data
        history_entry = {
            "timestamp": datetime.now().isoformat(),
            "symbol": symbol,
            "integrated_signal": {
                "direction": integrated_signal.get("direction", "NEUTRAL"),
                "confidence": integrated_signal.get("confidence", 0.0),
                "strength": integrated_signal.get("strength", 0)
            },
            "ai_prediction": {
                "available": ai_prediction.get("success", False),
                "direction": ai_prediction.get("prediction", {}).get("direction", "NEUTRAL"),
                "confidence": ai_prediction.get("prediction", {}).get("confidence", 0.0)
            },
            "technical_score": {
                "score": tech_signals.get("score", {}).get("score", 50),
                "direction": tech_signals.get("score", {}).get("direction", "NEUTRAL")
            }
        }
        
        # Add to history
        self.signal_history.append(history_entry)
        
        # Limit history size
        if len(self.signal_history) > self.max_history_size:
            self.signal_history = self.signal_history[-self.max_history_size:]
    
    def update_performance(self, symbol: str, signal_timestamp: str, 
                        actual_outcome: str, pnl: float) -> Dict:
        """
        Update strategy performance based on actual trade outcome
        
        Args:
            symbol: Trading symbol
            signal_timestamp: Timestamp of the original signal
            actual_outcome: Actual market outcome ("BUY", "SELL", "NEUTRAL")
            pnl: Profit/loss from the trade
            
        Returns:
            Updated performance metrics
        """
        # Find the original signal in history
        signal_entry = None
        for entry in self.signal_history:
            if (entry["timestamp"] == signal_timestamp and 
                entry["symbol"] == symbol):
                signal_entry = entry
                break
        
        if not signal_entry:
            return {
                "success": False,
                "error": f"Signal with timestamp {signal_timestamp} not found"
            }
        
        # Check correctness for each strategy
        integrated_correct = signal_entry["integrated_signal"]["direction"] == actual_outcome
        
        # Update AI prediction performance if available
        if signal_entry["ai_prediction"]["available"]:
            ai_correct = signal_entry["ai_prediction"]["direction"] == actual_outcome
            
            if ai_correct:
                self.strategy_performance["ai_prediction"]["correct"] += 1
            else:
                self.strategy_performance["ai_prediction"]["incorrect"] += 1
                
            self.strategy_performance["ai_prediction"]["total_pnl"] += pnl
        
        # Update technical strategy performance
        tech_correct = signal_entry["technical_score"]["direction"] == actual_outcome
        
        if tech_correct:
            self.strategy_performance["technical_bounce"]["correct"] += 1
            self.strategy_performance["support_resistance"]["correct"] += 1
            self.strategy_performance["volume_profile"]["correct"] += 1
        else:
            self.strategy_performance["technical_bounce"]["incorrect"] += 1
            self.strategy_performance["support_resistance"]["incorrect"] += 1
            self.strategy_performance["volume_profile"]["incorrect"] += 1
        
        # Update PnL
        self.strategy_performance["technical_bounce"]["total_pnl"] += pnl
        self.strategy_performance["support_resistance"]["total_pnl"] += pnl
        self.strategy_performance["volume_profile"]["total_pnl"] += pnl
        
        # Recalculate strategy weights based on performance
        self._update_strategy_weights()
        
        # Return updated metrics
        return {
            "success": True,
            "integrated_correct": integrated_correct,
            "updated_weights": self.strategy_weights,
            "performance_metrics": self.get_performance_metrics()
        }
    
    def _update_strategy_weights(self) -> None:
        """Update strategy weights based on performance metrics"""
        # Calculate accuracy for each strategy
        accuracies = {}
        
        for strategy, metrics in self.strategy_performance.items():
            total = metrics["correct"] + metrics["incorrect"]
            if total > 0:
                accuracies[strategy] = metrics["correct"] / total
            else:
                accuracies[strategy] = 0.0
        
        # Only update if we have enough data
        min_trades = 10
        if all(metrics["correct"] + metrics["incorrect"] >= min_trades 
               for metrics in self.strategy_performance.values()):
            
            # Calculate new weights
            total_accuracy = sum(accuracies.values())
            
            if total_accuracy > 0:
                # Base weights on accuracy with some smoothing
                new_weights = {}
                for strategy, accuracy in accuracies.items():
                    # Ensure minimum weight of 5%
                    new_weights[strategy] = max(0.05, accuracy / total_accuracy)
                
                # Normalize to sum to 1
                weight_sum = sum(new_weights.values())
                for strategy in new_weights:
                    new_weights[strategy] /= weight_sum
                
                # Update weights with some momentum (70% old, 30% new)
                for strategy in self.strategy_weights:
                    if strategy in new_weights:
                        self.strategy_weights[strategy] = (
                            0.7 * self.strategy_weights[strategy] + 
                            0.3 * new_weights[strategy]
                        )
                
                logger.info(f"Updated strategy weights: {self.strategy_weights}")
    
    def get_performance_metrics(self) -> Dict:
        """
        Get performance metrics for all strategies
        
        Returns:
            Dictionary with performance metrics
        """
        metrics = {}
        
        for strategy, perf in self.strategy_performance.items():
            total = perf["correct"] + perf["incorrect"]
            
            if total > 0:
                accuracy = perf["correct"] / total
                avg_pnl = perf["total_pnl"] / total
            else:
                accuracy = 0.0
                avg_pnl = 0.0
            
            metrics[strategy] = {
                "accuracy": accuracy,
                "correct": perf["correct"],
                "incorrect": perf["incorrect"],
                "total_trades": total,
                "total_pnl": perf["total_pnl"],
                "avg_pnl": avg_pnl,
                "weight": self.strategy_weights.get(strategy, 0.0)
            }
        
        return {
            "strategy_metrics": metrics,
            "total_signals": len(self.signal_history)
        }
    
    def save_state(self) -> bool:
        """
        Save the current state to disk
        
        Returns:
            Success flag
        """
        try:
            # Save signal history
            history_path = os.path.join(self.output_dir, "signal_history.json")
            
            with open(history_path, 'w') as f:
                import json
                json.dump(self.signal_history, f, indent=2)
            
            # Save performance metrics and weights
            metrics_path = os.path.join(self.output_dir, "strategy_metrics.json")
            
            metrics_data = {
                "performance": self.strategy_performance,
                "weights": self.strategy_weights,
                "timestamps": {
                    "saved_at": datetime.now().isoformat()
                }
            }
            
            with open(metrics_path, 'w') as f:
                import json
                json.dump(metrics_data, f, indent=2)
            
            logger.info("Strategy integrator state saved successfully")
            return True
            
        except Exception as e:
            logger.error(f"Error saving strategy integrator state: {str(e)}")
            return False
    
    def load_state(self) -> bool:
        """
        Load state from disk
        
        Returns:
            Success flag
        """
        try:
            # Load signal history
            history_path = os.path.join(self.output_dir, "signal_history.json")
            
            if os.path.exists(history_path):
                with open(history_path, 'r') as f:
                    import json
                    self.signal_history = json.load(f)
            
            # Load performance metrics and weights
            metrics_path = os.path.join(self.output_dir, "strategy_metrics.json")
            
            if os.path.exists(metrics_path):
                with open(metrics_path, 'r') as f:
                    import json
                    metrics_data = json.load(f)
                
                if "performance" in metrics_data:
                    self.strategy_performance = metrics_data["performance"]
                
                if "weights" in metrics_data:
                    self.strategy_weights = metrics_data["weights"]
            
            logger.info("État de l'intégrateur de stratégies chargé avec succès")
            return True
            
        except Exception as e:
            logger.error(f"Erreur lors du chargement de l'état: {str(e)}")
            return False