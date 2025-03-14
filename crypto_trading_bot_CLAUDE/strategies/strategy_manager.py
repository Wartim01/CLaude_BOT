"""
Strategy manager that combines and manages multiple trading strategies
"""
import importlib
import os
import json
import pandas as pd
from typing import Dict, List, Optional, Any
from datetime import datetime

from utils.logger import setup_logger

logger = setup_logger("strategy_manager")

class StrategyManager:
    """
    Manages multiple trading strategies and combines their signals
    """
    def __init__(self, config, data_fetcher, market_analyzer, scoring_engine):
        """
        Initialize the strategy manager
        
        Args:
            config: Configuration dictionary
            data_fetcher: Data fetcher instance
            market_analyzer: Market analyzer instance
            scoring_engine: Scoring engine instance
        """
        self.config = config
        self.data_fetcher = data_fetcher
        self.market_analyzer = market_analyzer
        self.scoring_engine = scoring_engine
        self.active_strategies = []
        self.strategy_weights = {}
        self.strategy_params = {}
        
        # Strategy performance metrics
        self.performance_metrics = {}
        
        # Load active strategies from config strategies["active"]
        active_list = self.config.get("strategies", {}).get("active", [])
        self.load_strategies(active_list)
    
    def load_strategies(self, strategy_names: list) -> None:
        """
        Charge dynamiquement les stratégies actives à partir de la liste de noms.
        """
        import importlib
        for strat_name in strategy_names:
            try:
                # Exemple : "trend_following" -> module "strategies.trend_following"
                module_path = f"strategies.{strat_name}"
                module = importlib.import_module(module_path)
                # Supposons que la classe s'appelle CamelCase version (ex: "TrendFollowing")
                class_name = "".join(word.capitalize() for word in strat_name.split("_"))
                strategy_class = getattr(module, class_name)
                # Instancier la stratégie et l’ajouter à la liste active
                instance = strategy_class(self.data_fetcher, self.market_analyzer, self.scoring_engine)
                self.active_strategies.append(instance)
            except Exception as e:
                logger.error(f"Failed to load strategy {strat_name}: {str(e)}")
    
    def run_strategies(self, symbol: str) -> Optional[Dict]:
        """
        Exécute toutes les stratégies actives pour le symbole donné et agrège
        leurs signaux via une combinaison pondérée.
        """
        signals = []
        for strategy in self.active_strategies:
            opp = strategy.find_trading_opportunity(symbol)
            if opp:
                signals.append(opp)
        if not signals:
            return None
        total_score = 0
        total_weight = 0
        vote = {"BUY": 0, "SELL": 0, "NEUTRAL": 0}
        for sig in signals:
            score = sig.get("score", 0)
            confidence = sig.get("confidence", 0.5)
            weight = confidence  # utilise la confiance comme poids
            total_score += score * weight
            total_weight += weight
            s = sig.get("signal", "NEUTRAL")
            if s in vote:
                vote[s] += weight
        avg_score = total_score / total_weight if total_weight else 0
        final_signal = max(vote, key=vote.get)
        if avg_score < self.config.get("strategies", {}).get("min_score", 50):
            final_signal = "NEUTRAL"
        return {"signal": final_signal, "avg_score": avg_score, "detailed_votes": vote}

    def generate_signal(self, symbol: str) -> Optional[Dict]:
        """
        Generate a trading signal from a single strategy
        
        Args:
            symbol: Trading pair symbol
            
        Returns:
            Trading signal dictionary
        """
        return self.run_strategies(symbol)
    
    def combine_signals(self, signals: Dict[str, Dict]) -> Dict:
        """
        Combine signals from multiple strategies
        
        Args:
            signals: Dictionary of strategy signals
            
        Returns:
            Combined signal
        """
        if not signals:
            return {"direction": "NEUTRAL", "signal_strength": 0}
        
        # Convert signals to numerical values
        signal_values = []
        signal_strengths = []
        signal_confidences = []
        timeframes = []
        
        for strategy_name, signal in signals.items():
            # Get direction value (-1 for SELL, 0 for NEUTRAL, 1 for BUY)
            direction = signal.get("direction", "NEUTRAL")
            direction_value = 0
            
            if direction == "BUY":
                direction_value = 1
            elif direction == "SELL":
                direction_value = -1
            
            # Get signal strength (0-5)
            strength = signal.get("signal_strength", 0)
            
            # Get confidence (0-1)
            confidence = signal.get("confidence", 0.5)
            
            # Get timeframe
            if "timeframe" in signal:
                timeframes.append(signal["timeframe"])
            
            # Get strategy weight (default to 1)
            weight = self.strategy_weights.get(strategy_name, 1)
            
            # Add to lists
            signal_values.append(direction_value * weight)
            signal_strengths.append(strength * weight)
            signal_confidences.append(confidence * weight)
        
        # Calculate weighted values
        total_weights = sum(self.strategy_weights.get(name, 1) for name in signals.keys())
        
        if total_weights > 0:
            weighted_value = sum(signal_values) / total_weights
            weighted_strength = sum(signal_strengths) / total_weights
            weighted_confidence = sum(signal_confidences) / total_weights
        else:
            weighted_value = sum(signal_values) / len(signal_values)
            weighted_strength = sum(signal_strengths) / len(signal_strengths)
            weighted_confidence = sum(signal_confidences) / len(signal_confidences)
        
        # Determine combined signal direction
        if weighted_value > 0.2:
            direction = "BUY"
        elif weighted_value < -0.2:
            direction = "SELL"
        else:
            direction = "NEUTRAL"
        
        # Create combined signal
        combined_signal = {
            "direction": direction,
            "signal_strength": abs(int(weighted_strength)),
            "confidence": float(weighted_confidence),
            "weighted_value": float(weighted_value),
            "signals": list(signals.keys()),
            "timestamp": datetime.now().isoformat()
        }
        
        # Add timeframe if possible
        if timeframes:
            combined_signal["timeframe"] = max(set(timeframes), key=timeframes.count)
        
        return combined_signal
    
    def check_exit(self, symbol: str, direction: str, data: pd.DataFrame, position: Dict) -> Dict:
        """
        Check if a position should be exited
        
        Args:
            symbol: Trading pair symbol
            direction: Current position direction ('BUY' or 'SELL')
            data: Market data DataFrame
            position: Position data
            
        Returns:
            Exit signal dict
        """
        exit_signals = {}
        
        # Check each strategy for exit signals
        for name, strategy in self.active_strategies:
            try:
                exit_signal = None
                
                # Handle both class and functional strategies
                if hasattr(strategy, 'check_exit'):
                    # Class-based strategy
                    exit_signal = strategy.check_exit(data, symbol, direction, position)
                elif isinstance(strategy, dict) and "module" in strategy:
                    # Functional strategy
                    module = strategy["module"]
                    
                    if hasattr(module, 'check_exit'):
                        exit_signal = module.check_exit(data, symbol, direction, position, **strategy["params"])
                
                if exit_signal and exit_signal.get("should_exit", False):
                    exit_signals[name] = exit_signal
            except Exception as e:
                logger.error(f"Error checking exit with strategy {name}: {str(e)}")
        
        # Combine exit signals (exit if any strategy says to)
        if exit_signals:
            # Count exit signals
            exit_count = len(exit_signals)
            total_count = len(self.active_strategies)
            
            # Exit if more than half of strategies say to exit
            should_exit = exit_count > total_count / 2
            
            # Collect reasons
            reasons = [signal.get("reason", "Strategy exit") for signal in exit_signals.values()]
            
            return {
                "should_exit": should_exit,
                "reason": f"{exit_count}/{total_count} strategies recommend exit: {', '.join(reasons[:2])}",
                "exit_signals": exit_signals
            }
        
        return {"should_exit": False}
    
    def update_performance(self, symbol: str, signal_timestamp: str, actual_outcome: str, pnl: float) -> None:
        """
        Update strategy performance metrics
        
        Args:
            symbol: Trading pair symbol
            signal_timestamp: Timestamp of the signal
            actual_outcome: Actual trade outcome ('BUY', 'SELL', 'NEUTRAL')
            pnl: Profit/Loss from the trade
        """
        # For each strategy, check if it generated a correct signal
        for name, strategy in self.active_strategies:
            # Get the last signal from this strategy for the symbol
            signal_direction = None
            
            if hasattr(strategy, 'get_last_signal'):
                # Class-based strategy
                last_signal = strategy.get_last_signal(symbol)
                if last_signal:
                    signal_direction = last_signal.get("direction")
            
            if signal_direction and signal_direction != "NEUTRAL":
                # Update metrics
                self.performance_metrics[name]["total_pnl"] += pnl
                
                # Check if the signal was correct
                if (signal_direction == "BUY" and actual_outcome == "BUY") or \
                   (signal_direction == "SELL" and actual_outcome == "SELL"):
                    self.performance_metrics[name]["correct_signals"] += 1
                else:
                    self.performance_metrics[name]["incorrect_signals"] += 1
                
                # Update win rate
                total_signals = self.performance_metrics[name]["correct_signals"] + self.performance_metrics[name]["incorrect_signals"]
                if total_signals > 0:
                    self.performance_metrics[name]["win_rate"] = (
                        self.performance_metrics[name]["correct_signals"] / total_signals * 100
                    )
                
                # Update average PnL
                if self.performance_metrics[name]["signals_generated"] > 0:
                    self.performance_metrics[name]["avg_pnl_per_trade"] = (
                        self.performance_metrics[name]["total_pnl"] / self.performance_metrics[name]["signals_generated"]
                    )
    
    def get_performance_metrics(self) -> Dict:
        """
        Get performance metrics for all strategies
        
        Returns:
            Dictionary of strategy performance metrics
        """
        # Overall metrics
        total_signals = sum(metrics["signals_generated"] for metrics in self.performance_metrics.values())
        total_correct = sum(metrics["correct_signals"] for metrics in self.performance_metrics.values())
        total_pnl = sum(metrics["total_pnl"] for metrics in self.performance_metrics.values())
        
        overall_win_rate = 0
        if total_signals > 0:
            overall_win_rate = total_correct / total_signals * 100
        
        return {
            "overall": {
                "total_signals": total_signals,
                "win_rate": overall_win_rate,
                "total_pnl": total_pnl
            },
            "strategies": self.performance_metrics
        }