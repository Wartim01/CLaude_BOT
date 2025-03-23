"""
Strategy manager for the trading bot
Manages and coordinates multiple trading strategies
"""
import logging
from typing import Dict, List, Optional, Any
import importlib

from utils.logger import setup_logger
from config.trading_params import MINIMUM_SCORE_TO_TRADE

logger = setup_logger("strategy_manager")

class StrategyManager:
    """
    Manages all trading strategies, coordinates their execution,
    and aggregates their signals
    """
    def __init__(self):
        """Initialize the strategy manager"""
        self.strategies = {}
        self.active_strategies = []
        self.strategy_weights = {}
        self.config = {}
        self.technical_bounce_strategy = None  # Will be set if that strategy is loaded
        
        # Load strategies from configuration
        self._load_strategies()
        
    def _load_strategies(self):
        """Load strategies based on configuration"""
        try:
            # This is a simplified implementation; normally you would:
            # 1. Read strategy configurations from config file
            # 2. Dynamically import and instantiate strategies
            # 3. Set weights based on configuration
            
            # For now, we'll just set up a simple structure with placeholder strategies
            from strategies.technical_bounce import TechnicalBounceStrategy
            
            # Mock dependencies that would normally be injected
            class MockDataFetcher:
                def get_market_data(self, symbol):
                    return {}
                    
                def detect_volume_spike(self, symbol):
                    return {"spike": False, "ratio": 1.0}
            
            class MockMarketAnalyzer:
                def analyze_market_state(self, symbol):
                    return {"favorable": True}
            
            class MockScoringEngine:
                def calculate_opportunity_score(self, factors):
                    # Simple weighted average of factors
                    weights = {
                        "rsi_level": 0.3,
                        "bb_position": 0.3,
                        "trend_strength": 0.2,
                        "volume_strength": 0.2
                    }
                    score = 0
                    for factor, value in factors.items():
                        if factor in weights:
                            score += value * weights[factor]
                    return score * 100  # Scale to 0-100
            
            # Create technical bounce strategy
            self.technical_bounce_strategy = TechnicalBounceStrategy(
                MockDataFetcher(),
                MockMarketAnalyzer(),
                MockScoringEngine()
            )
            
            self.strategies["technical_bounce"] = self.technical_bounce_strategy
            self.active_strategies.append("technical_bounce")
            self.strategy_weights["technical_bounce"] = 1.0
            
            logger.info(f"Loaded strategies: {', '.join(self.active_strategies)}")
            logger.info(f"Technical bounce strategy min score: {self.technical_bounce_strategy.min_score}")
            
        except Exception as e:
            logger.error(f"Error loading strategies: {str(e)}")
    
    def run_strategies(self, data: Any, symbol: str, timeframe: str) -> Dict:
        """
        Run all active strategies and aggregate results
        
        Args:
            data: Market data
            symbol: Trading pair symbol
            timeframe: Timeframe
            
        Returns:
            Aggregated strategy results
        """
        signals = {}
        
        for strategy_name in self.active_strategies:
            try:
                strategy = self.strategies[strategy_name]
                result = strategy.find_trading_opportunity(symbol)
                
                if result:
                    signals[strategy_name] = result
            except Exception as e:
                logger.error(f"Error running strategy {strategy_name}: {str(e)}")
        
        # Aggregate signals (simple implementation)
        if not signals:
            return {"signal": "NEUTRAL", "score": 0, "details": {}}
        
        # If we have at least one signal, use the highest scoring one
        best_strategy = max(signals.items(), key=lambda x: x[1]["score"])
        strategy_name, best_signal = best_strategy
        
        return {
            "signal": "BUY" if best_signal["score"] >= MINIMUM_SCORE_TO_TRADE else "NEUTRAL",
            "score": best_signal["score"],
            "strategy": strategy_name,
            "details": best_signal
        }
    
    def update_strategy_thresholds(self, new_threshold: float) -> None:
        """
        Update thresholds for all strategies that support it
        
        Args:
            new_threshold: New threshold value to set
        """
        updated_count = 0
        
        # Update technical bounce strategy if available
        if self.technical_bounce_strategy:
            try:
                old_threshold = self.technical_bounce_strategy.min_score
                self.technical_bounce_strategy.update_min_score(new_threshold)
                logger.info(f"Updated technical_bounce strategy threshold: {old_threshold} -> {new_threshold}")
                updated_count += 1
            except Exception as e:
                logger.error(f"Failed to update technical_bounce threshold: {str(e)}")
        
        # Add other strategy updates here when available
        
        logger.info(f"Updated thresholds for {updated_count} strategies")
        return updated_count
    
    def get_opportunity_statistics(self, lookback_hours: int = 24) -> List[Dict]:
        """
        Get opportunity statistics from all strategies that support it
        
        Args:
            lookback_hours: Hours to look back for statistics
            
        Returns:
            List of statistics dictionaries from supporting strategies
        """
        stats_list = []
        
        # Get stats from technical bounce strategy
        if self.technical_bounce_strategy and hasattr(self.technical_bounce_strategy, "get_opportunities_stats"):
            try:
                stats = self.technical_bounce_strategy.get_opportunities_stats(lookback_hours=lookback_hours)
                stats_list.append(stats)
                logger.debug(f"Got opportunity statistics from technical_bounce: {stats}")
            except Exception as e:
                logger.error(f"Error getting opportunity stats from technical_bounce: {str(e)}")
        
        # Add other strategy stats collection here when available
        
        return stats_list