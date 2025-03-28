"""
Risk management system for controlling trading exposure and preventing large losses
"""
import os
import json
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta

from utils.logger import setup_logger, log_threshold_adjustment, log_adjustment_history
from config.config import DATA_DIR
from utils.correlation_matrix import CorrelationMatrix
from config.trading_params import MAX_CONCURRENT_TRADES

logger = setup_logger("risk_manager")

class RiskManager:
    """
    Manages trading risk by controlling position sizes, monitoring exposure,
    and adaptively adjusting decision thresholds
    """
    def __init__(self, capital: float, kelly_fraction: float = 0.3, max_risk_per_trade: float = 5.0,
               default_risk_per_trade: float = 0.02, 
               account_risk_limit: float = 0.10,
               max_trades_per_symbol: int = MAX_CONCURRENT_TRADES,  # Use global parameter as default
               use_correlation_adjustment: bool = True,
               threshold_config: dict = None,
               base_threshold: float = 70.0,
               notification_handler=None,
               config=None):
        """
        Initialize the risk manager
        
        Args:
            capital: Initial capital
            kelly_fraction: Fraction of Kelly criterion to use
            max_risk_per_trade: Maximum risk per trade (as percentage of capital)
            default_risk_per_trade: Default risk per trade (as decimal percentage of account)
            account_risk_limit: Maximum total risk exposure (as decimal percentage of account)
            max_trades_per_symbol: Maximum number of concurrent trades per symbol
            use_correlation_adjustment: Whether to adjust risk based on correlation between assets
            threshold_config: Configuration for adaptive threshold adjustment
            base_threshold: Base threshold value for trading decisions
            notification_handler: Handler for sending notifications
            config: Additional configuration parameters
        """
        # Existing initialization
        self.capital = capital
        self.kelly_fraction = kelly_fraction
        self.max_risk_per_trade = max_risk_per_trade / 100.0  # Convert from percentage
        self.default_risk_per_trade = default_risk_per_trade
        self.account_risk_limit = account_risk_limit
        self.max_trades_per_symbol = max_trades_per_symbol
        self.use_correlation_adjustment = use_correlation_adjustment
        self.config = config or {}
        
        # Initialize risk tracking
        self.active_positions = {}
        self.historical_positions = []
        self.risk_history = []
        self.risk_per_symbol = {}
        self.total_exposure = 0.0
        self.total_risk_amount = 0.0
        self.initial_balance = capital
        # Initialize market cooldown state if not defined
        self.market_cooldown = {
            "active": False,
            "start_time": None,
            "reason": None,
            "duration": timedelta(minutes=60)  # using MARKET_COOLDOWN_PERIOD from trading_params
        }
        # Set a default win rate
        self.win_rate = 50.0
        # Set default volatility thresholds
        self.volatility_thresholds = {
            "extreme": 5.0,
            "high": 3.0
        }
        
        # Path for storing risk data
        self.data_path = os.path.join(DATA_DIR, "risk", "risk_manager_state.json")
        os.makedirs(os.path.dirname(self.data_path), exist_ok=True)
        
        # Load previous state if available
        self._load_state()
        
        # Notification handler for alerts
        self.notification_handler = notification_handler
        
        # Initialize threshold adjustment system
        self._init_threshold_adjustment(threshold_config, base_threshold)
        
        logger.info(f"Risk manager initialized with capital: {capital}, max risk: {account_risk_limit*100}%, threshold: {self.threshold_adjustment['current_threshold']}")

    def _init_threshold_adjustment(self, threshold_config: dict, base_threshold: float) -> None:
        """
        Initialize the threshold adjustment system
        
        Args:
            threshold_config: Configuration for adaptive threshold adjustment
            base_threshold: Base threshold value for trading decisions
        """
        # Default configuration if none provided
        default_config = {
            "enabled": True,
            "check_interval_hours": 6,
            "adjustment_step": 2.0,
            "max_adjustment": 10.0,
            "min_threshold": 62.0,
            "max_threshold": 85.0,
            "post_adjustment_cooldown_hours": 12
        }
        
        # Use provided config or default
        config = threshold_config or default_config
        
        # Setup threshold adjustment tracking
        self.threshold_adjustment = {
            "enabled": config.get("enabled", True),
            "last_check": datetime.now(),
            "check_interval_hours": config.get("check_interval_hours", 6),
            "adjustment_step": config.get("adjustment_step", 2.0),
            "max_adjustment": config.get("max_adjustment", 10.0),
            "min_threshold": config.get("min_threshold", 62.0),
            "max_threshold": config.get("max_threshold", 85.0),
            "base_threshold": base_threshold,
            "current_threshold": base_threshold,
            "adjustment_history": [],
            "last_adjustment_time": None,
            "post_adjustment_cooldown_hours": config.get("post_adjustment_cooldown_hours", 12)
        }
        
        logger.info(f"Threshold adjustment system initialized. Base: {base_threshold}, Min: {self.threshold_adjustment['min_threshold']}, Max: {self.threshold_adjustment['max_threshold']}")

    # ===== Threshold Adjustment Methods =====
    
    def get_current_threshold(self) -> float:
        """
        Get the current decision threshold
        
        Returns:
            Current threshold value
        """
        return self.threshold_adjustment["current_threshold"]
    
    def check_threshold_adjustment(self, opportunity_stats: List[Dict] = None, recent_trades: List[Dict] = None) -> Dict:
        """
        Check if threshold adjustment is needed and apply if necessary
        
        Args:
            opportunity_stats: Statistics about trading opportunities
            recent_trades: List of recent trades
            
        Returns:
            Adjustment result dictionary
        """
        if not self.threshold_adjustment["enabled"]:
            return {"adjusted": False, "reason": "Threshold adjustment disabled"}
        
        current_time = datetime.now()
        last_check = self.threshold_adjustment["last_check"]
        check_interval = timedelta(hours=self.threshold_adjustment["check_interval_hours"])
        
        # Only check periodically
        if current_time - last_check < check_interval:
            return {"adjusted": False, "reason": "Not time for check yet"}
        
        # Check if we're still in cooldown period after last adjustment
        last_adjustment_time = self.threshold_adjustment.get("last_adjustment_time")
        cooldown_hours = self.threshold_adjustment["post_adjustment_cooldown_hours"]
        
        if last_adjustment_time and (current_time - last_adjustment_time < timedelta(hours=cooldown_hours)):
            logger.debug(f"Threshold adjustment in cooldown period. Next check possible after {last_adjustment_time + timedelta(hours=cooldown_hours)}")
            return {"adjusted": False, "reason": "In cooldown period"}
        
        # Update last check time
        self.threshold_adjustment["last_check"] = current_time
        
        # Analyze opportunities and make adjustment decision
        return self._analyze_and_adjust_thresholds(opportunity_stats, recent_trades)
        
    def _analyze_and_adjust_thresholds(self, stats_list: List[Dict], recent_trades: List[Dict]) -> Dict:
        """
        Analyze opportunity statistics and adjust thresholds if necessary
        
        Args:
            stats_list: List of opportunity statistics from different strategies
            recent_trades: List of recent trades
            
        Returns:
            Adjustment result dictionary
        """
        if not stats_list:
            logger.debug("No opportunity statistics provided for threshold analysis")
            return {"adjusted": False, "reason": "No statistics provided"}
            
        # Calculate combined statistics
        total_opportunities = sum(stats.get("total_opportunities", 0) for stats in stats_list)
        executed_opportunities = sum(stats.get("executed_opportunities", 0) for stats in stats_list)
        missed_opportunities = sum(stats.get("missed_opportunities", 0) for stats in stats_list)
        near_misses = sum(stats.get("near_misses", 0) for stats in stats_list)
        
        # If no opportunities detected at all, consider lowering the threshold
        if total_opportunities == 0:
            result = self._adjust_threshold_lower("No trading opportunities detected")
            return {"adjusted": result["adjusted"], "new_threshold": result["threshold"], "reason": result["reason"]}
            
        # Check recency of trades
        has_recent_trades = False
        if recent_trades:
            # Get timestamp of most recent trade
            most_recent = max((trade.get("close_time", datetime.now() - timedelta(days=7)) for trade in recent_trades), 
                              default=datetime.now() - timedelta(days=7))
            has_recent_trades = (datetime.now() - most_recent) < timedelta(hours=24)
        
        # If we have too many missed opportunities, consider lowering the threshold
        if missed_opportunities > 0 and executed_opportunities == 0:
            result = self._adjust_threshold_lower("Only missed opportunities, no executions")
            return {"adjusted": result["adjusted"], "new_threshold": result["threshold"], "reason": result["reason"]}
            
        # If we have near misses but few executions, consider lowering the threshold slightly
        if near_misses > 5 and executed_opportunities < 2:
            result = self._adjust_threshold_lower("Multiple near-miss opportunities but few executions")
            return {"adjusted": result["adjusted"], "new_threshold": result["threshold"], "reason": result["reason"]}
            
        # If we have many executed opportunities, consider raising the threshold for better quality
        if executed_opportunities > 10:
            result = self._adjust_threshold_higher("Many opportunities being executed, increasing quality filter")
            return {"adjusted": result["adjusted"], "new_threshold": result["threshold"], "reason": result["reason"]}
            
        # If we have a good balance (some misses, some executions), keep current threshold
        logger.info(f"Threshold analysis: found {executed_opportunities} executed and {missed_opportunities} missed opportunities. Current threshold of {self.threshold_adjustment['current_threshold']} appears balanced.")
        return {"adjusted": False, "reason": "Threshold appears balanced", "new_threshold": self.threshold_adjustment["current_threshold"]}
    
    def _adjust_threshold_lower(self, reason: str) -> Dict:
        """
        Lower the minimum score threshold to detect more opportunities
        
        Args:
            reason: Reason for adjustment
            
        Returns:
            Adjustment result dictionary
        """
        current = self.threshold_adjustment["current_threshold"]
        step = self.threshold_adjustment["adjustment_step"]
        min_threshold = self.threshold_adjustment["min_threshold"]
        base_threshold = self.threshold_adjustment["base_threshold"]
        max_adjustment = self.threshold_adjustment["max_adjustment"]
        
        # Calculate maximum allowed adjustment down from base
        max_down = min(base_threshold - min_threshold, max_adjustment)
        
        # Check if we're already at the minimum allowed
        if current <= min_threshold or current <= (base_threshold - max_down):
            logger.info(f"Cannot lower threshold below minimum ({min_threshold})")
            return {"adjusted": False, "threshold": current, "reason": f"Already at minimum ({min_threshold})"}
        
        # Calculate new threshold
        new_threshold = max(min_threshold, current - step)
        
        # Apply the adjustment
        result = self._apply_threshold_adjustment(new_threshold, reason, "decrease")
        return {"adjusted": True, "threshold": new_threshold, "reason": reason}
    
    def _adjust_threshold_higher(self, reason: str) -> Dict:
        """
        Raise the minimum score threshold for better quality trades
        
        Args:
            reason: Reason for adjustment
            
        Returns:
            Adjustment result dictionary
        """
        current = self.threshold_adjustment["current_threshold"]
        step = self.threshold_adjustment["adjustment_step"]
        max_threshold = self.threshold_adjustment["max_threshold"]
        base_threshold = self.threshold_adjustment["base_threshold"]
        max_adjustment = self.threshold_adjustment["max_adjustment"]
        
        # Calculate maximum allowed adjustment up from base
        max_up = min(max_threshold - base_threshold, max_adjustment)
        
        # Check if we're already at the maximum allowed
        if current >= max_threshold or current >= (base_threshold + max_up):
            logger.info(f"Cannot raise threshold above maximum ({max_threshold})")
            return {"adjusted": False, "threshold": current, "reason": f"Already at maximum ({max_threshold})"}
        
        # Calculate new threshold
        new_threshold = min(max_threshold, current + step)
        
        # Apply the adjustment
        result = self._apply_threshold_adjustment(new_threshold, reason, "increase")
        return {"adjusted": True, "threshold": new_threshold, "reason": reason}
    
    def _apply_threshold_adjustment(self, new_threshold: float, reason: str, direction: str) -> Dict:
        """
        Apply a threshold adjustment and record the change
        
        Args:
            new_threshold: New threshold value
            reason: Reason for adjustment
            direction: 'increase' or 'decrease'
            
        Returns:
            Adjustment result dictionary
        """
        old_threshold = self.threshold_adjustment["current_threshold"]
        
        # Update our tracking
        self.threshold_adjustment["current_threshold"] = new_threshold
        self.threshold_adjustment["last_adjustment_time"] = datetime.now()
        
        # Record adjustment in history
        adjustment = {
            "timestamp": datetime.now().isoformat(),
            "old_threshold": old_threshold,
            "new_threshold": new_threshold,
            "direction": direction,
            "reason": reason
        }
        
        self.threshold_adjustment["adjustment_history"].append(adjustment)
        
        # Trim history if needed
        if len(self.threshold_adjustment["adjustment_history"]) > 50:
            self.threshold_adjustment["adjustment_history"] = self.threshold_adjustment["adjustment_history"][-50:]
        
        # Log the adjustment using the special logger function
        change = abs(new_threshold - old_threshold)
        log_threshold_adjustment(logger, old_threshold, new_threshold, reason)
        
        # Save adjustment history to disk periodically
        if len(self.threshold_adjustment["adjustment_history"]) % 5 == 0:
            log_adjustment_history(self.threshold_adjustment["adjustment_history"])
        
        # Send notification if handler is available
        if self.notification_handler:
            self.notification_handler.send_message(
                "Trading Bot - Strategy Adjustment",
                f"Trading threshold {direction}d by {change:.2f} points to {new_threshold:.1f}.\n"
                f"Reason: {reason}"
            )
            
        # Save state to persist threshold changes
        self._save_state()
            
        return {
            "adjusted": True, 
            "old_threshold": old_threshold,
            "new_threshold": new_threshold,
            "direction": direction,
            "reason": reason
        }

    def update_strategy_thresholds(self, strategies_dict: Dict) -> None:
        """
        Update minimum score thresholds in all provided strategies
        
        Args:
            strategies_dict: Dictionary of strategy objects
        """
        current_threshold = self.threshold_adjustment["current_threshold"]
        
        for strategy_name, strategy_obj in strategies_dict.items():
            if hasattr(strategy_obj, "update_min_score"):
                try:
                    strategy_obj.update_min_score(current_threshold)
                    logger.debug(f"Updated threshold for {strategy_name} to {current_threshold}")
                except Exception as e:
                    logger.error(f"Failed to update threshold for {strategy_name}: {str(e)}")

    # Make sure we include threshold adjustment data in state saving/loading
    def _save_state(self) -> None:
        """Save risk manager state to disk"""
        state = {
            "active_positions": self.active_positions,
            "risk_history": self.risk_history,
            "initial_balance": self.initial_balance,
            "total_exposure": self.total_exposure,
            "total_risk_amount": self.total_risk_amount,
            "risk_per_symbol": self.risk_per_symbol,
            "threshold_adjustment": self.threshold_adjustment,  # Include threshold data
            "timestamp": datetime.now().isoformat()
        }
        
        try:
            with open(self.data_path, "w") as f:
                json.dump(state, f, indent=2, default=str)
        except Exception as e:
            logger.error(f"Error saving risk manager state: {e}")
    
    def _load_state(self) -> None:
        """Load risk manager state from disk"""
        if not os.path.exists(self.data_path):
            return
        
        try:
            with open(self.data_path, "r") as f:
                state = json.load(f)
            
            self.active_positions = state.get("active_positions", {})
            self.risk_history = state.get("risk_history", [])
            self.initial_balance = state.get("initial_balance")
            self.total_exposure = state.get("total_exposure", 0.0)
            self.total_risk_amount = state.get("total_risk_amount", 0.0)
            self.risk_per_symbol = state.get("risk_per_symbol", {})
            
            # Load threshold adjustment data if available
            if "threshold_adjustment" in state:
                # Preserve basic configuration but load current state
                if hasattr(self, "threshold_adjustment"):
                    base_config = self.threshold_adjustment
                    loaded_threshold = state["threshold_adjustment"]
                    
                    # Only update keys that represent state, not configuration
                    for key in ["current_threshold", "adjustment_history", "last_adjustment_time", "last_check"]:
                        if key in loaded_threshold:
                            base_config[key] = loaded_threshold[key]
                    
                    self.threshold_adjustment = base_config
                else:
                    self.threshold_adjustment = state["threshold_adjustment"]
            
            logger.info(f"Loaded risk manager state: {len(self.active_positions)} active positions, threshold: {self.threshold_adjustment.get('current_threshold')}")
        except Exception as e:
            logger.error(f"Error loading risk manager state: {e}")

    def evaluate_trade_risk(self, symbol: str, direction: str, entry_price: float,
                          stop_loss: float, account_balance: float,
                          market_data: Optional[Dict[str, pd.DataFrame]] = None) -> Dict[str, Any]:
        """
        Evaluate whether a trade is within acceptable risk parameters
        
        Args:
            symbol: Trading symbol
            direction: Trade direction ('BUY' or 'SELL')
            entry_price: Proposed entry price
            stop_loss: Proposed stop loss price
            account_balance: Current account balance
            market_data: Optional dict of market data for correlation analysis
            
        Returns:
            Risk evaluation result dict
        """
        # First check if we're in a market cooldown period
        if self.market_cooldown["active"]:
            current_time = datetime.now()
            if current_time - self.market_cooldown["start_time"] <= self.market_cooldown["duration"]:
                # Still in cooldown period
                remaining = (self.market_cooldown["start_time"] + self.market_cooldown["duration"] - current_time).total_seconds() / 60
                return {
                    "approved": False,
                    "reason": f"Market in cooldown period ({remaining:.1f} minutes remaining). Reason: {self.market_cooldown['reason']}"
                }
            else:
                # Cooldown period expired, reset cooldown state
                self.market_cooldown["active"] = False
                self.market_cooldown["start_time"] = None
                self.market_cooldown["reason"] = None
                logger.info("Market cooldown period ended. Trading restrictions lifted.")
        
        # Store initial balance if not set
        if self.initial_balance is None:
            self.initial_balance = account_balance
        
        # Calculate base risk amount
        risk_amount = account_balance * self.default_risk_per_trade
        
        # Calculate stop distance and validate
        if direction == "BUY":
            if stop_loss >= entry_price:
                return {
                    "approved": False, 
                    "reason": f"Invalid stop loss ({stop_loss}) for BUY order (entry: {entry_price})"
                }
            stop_distance = (entry_price - stop_loss) / entry_price
        else:  # SELL
            if stop_loss <= entry_price:
                return {
                    "approved": False, 
                    "reason": f"Invalid stop loss ({stop_loss}) for SELL order (entry: {entry_price})"
                }
            stop_distance = (stop_loss - entry_price) / entry_price
        
        # Validate stop distance
        min_stop_distance = 0.005  # 0.5% minimum stop distance
        max_stop_distance = 0.15   # 15% maximum stop distance
        
        if stop_distance < min_stop_distance:
            return {
                "approved": False,
                "reason": f"Stop too close ({stop_distance:.2%}), minimum is {min_stop_distance:.2%}"
            }
        
        if stop_distance > max_stop_distance:
            return {
                "approved": False,
                "reason": f"Stop too far ({stop_distance:.2%}), maximum is {max_stop_distance:.2%}"
            }
        
        # Calculate position size based on risk
        position_size = risk_amount / (entry_price * stop_distance)
        position_value = position_size * entry_price
        
        # Check if we already have trades for this symbol
        symbol_trades = 0
        for _, position in self.active_positions.items():
            if position["symbol"] == symbol:
                symbol_trades += 1
        
        if symbol_trades >= self.max_trades_per_symbol:
            return {
                "approved": False,
                "reason": f"Maximum trades per symbol ({self.max_trades_per_symbol}) reached for {symbol}"
            }
        
        # Ensure compatibility with global MAX_CONCURRENT_TRADES parameter
        if self.max_trades_per_symbol != MAX_CONCURRENT_TRADES:
            logger.warning(f"Risk manager max_trades_per_symbol ({self.max_trades_per_symbol}) differs from global MAX_CONCURRENT_TRADES ({MAX_CONCURRENT_TRADES})")
        
        # Calculate current total risk
        current_risk = self.calculate_total_risk()
        
        # Calculate additional risk from this trade
        additional_risk = risk_amount / account_balance
        
        # Check if this would exceed our risk limit
        if current_risk + additional_risk > self.account_risk_limit:
            return {
                "approved": False,
                "reason": f"Risk limit exceeded ({current_risk+additional_risk:.2%} > {self.account_risk_limit:.2%})"
            }
        
        # Apply correlation adjustment if enabled and market data provided
        correlation_factor = 1.0
        if self.use_correlation_adjustment and market_data:
            # Initialize the correlation matrix and update with market data
            cm = CorrelationMatrix(cache_duration=3600)
            cm.update_matrix(market_data, time_window='7d')
            correlations = []
            # For each active position, get the correlation with the new symbol
            for pos in self.active_positions.values():
                pos_symbol = pos.get("symbol")
                if pos_symbol and pos_symbol != symbol:
                    corr = cm.get_correlation(symbol, pos_symbol, market_data)
                    if corr is not None and not np.isnan(corr):
                        correlations.append(corr)
            if correlations:
                avg_corr = np.mean(correlations)
                # Example: if average correlation is high, reduce risk (here assume >0.5 is high)
                correlation_factor = 1 - avg_corr if avg_corr > 0.5 else 1.0
        
        # Final position sizing and approval
        return {
            "approved": True,
            "position_sizing": {
                "position_size": position_size * correlation_factor,
                "position_value": position_size * entry_price * correlation_factor,
                "risk_amount": risk_amount,
                "risk_percentage": self.default_risk_per_trade * 100,
                "stop_distance_percentage": stop_distance * 100,
                "correlation_adjustment": correlation_factor
            },
            "risk_status": {
                "current_risk": current_risk,
                "additional_risk": additional_risk,
                "total_risk_after_trade": current_risk + additional_risk,
                "risk_limit": self.account_risk_limit
            }
        }
    
    def register_new_trade(self, trade_id: str, symbol: str, direction: str, 
                         entry_price: float, stop_loss: float, position_size: float,
                         risk_percentage: float, account_balance: float) -> Dict:
        """
        Register a new trade in the risk management system
        
        Args:
            trade_id: Unique trade identifier
            symbol: Trading symbol
            direction: Trade direction
            entry_price: Entry price
            stop_loss: Stop loss price
            position_size: Position size
            risk_percentage: Risk percentage (as decimal)
            account_balance: Current account balance
            
        Returns:
            Registration result
        """
        # Calculate risk amount
        risk_amount = account_balance * risk_percentage
        
        # Calculate position value
        position_value = entry_price * position_size
        
        # Add to active positions
        self.active_positions[trade_id] = {
            "id": trade_id,
            "symbol": symbol,
            "direction": direction,
            "entry_price": entry_price,
            "stop_loss": stop_loss,
            "position_size": position_size,
            "position_value": position_value,
            "risk_amount": risk_amount,
            "risk_percentage": risk_percentage,
            "entry_time": datetime.now().isoformat(),
            "last_update": datetime.now().isoformat(),
            "current_price": entry_price,
            "current_value": position_value,
            "unrealized_pnl": 0.0
        }
        
        # Update symbol-specific risk
        if symbol not in self.risk_per_symbol:
            self.risk_per_symbol[symbol] = 0.0
        
        self.risk_per_symbol[symbol] += risk_amount
        
        # Update metrics
        self.total_exposure += position_value
        self.total_risk_amount += risk_amount
        
        # Save state
        self._save_state()
        
        logger.info(f"Registered new trade {trade_id} for {symbol}, risk amount: {risk_amount:.2f}")
        return {"success": True, "trade_id": trade_id}
    
    def update_trade_status(self, trade_id: str, current_price: float, account_balance: float) -> Dict:
        """
        Update the status of an active trade
        
        Args:
            trade_id: Trade identifier
            current_price: Current market price
            account_balance: Current account balance
            
        Returns:
            Update result
        """
        if trade_id not in self.active_positions:
            return {"success": False, "reason": f"Trade {trade_id} not found"}
        
        position = self.active_positions[trade_id]
        
        # Calculate current value and PnL
        old_value = position["current_value"]
        new_value = current_price * position["position_size"]
        
        # Update position data
        position["current_price"] = current_price
        position["current_value"] = new_value
        
        # Calculate PnL based on direction
        if position["direction"] == "BUY":
            position["unrealized_pnl"] = new_value - (position["entry_price"] * position["position_size"])
        else:  # SELL
            position["unrealized_pnl"] = (position["entry_price"] * position["position_size"]) - new_value
        
        position["last_update"] = datetime.now().isoformat()
        
        # Update overall exposure
        self.total_exposure = self.total_exposure - old_value + new_value
        
        return {
            "success": True, 
            "value": new_value,
            "pnl": position["unrealized_pnl"]
        }
    
    def close_position(self, trade_id: str, exit_price: float, exit_reason: str) -> Dict:
        """
        Close a position and update risk metrics
        
        Args:
            trade_id: Trade identifier
            exit_price: Exit price
            exit_reason: Reason for closing the position
            
        Returns:
            Close result
        """
        if trade_id not in self.active_positions:
            return {"success": False, "reason": f"Trade {trade_id} not found"}
        
        position = self.active_positions[trade_id]
        symbol = position["symbol"]
        
        # Calculate realized PnL
        if position["direction"] == "BUY":
            realized_pnl = (exit_price - position["entry_price"]) * position["position_size"]
        else:  # SELL
            realized_pnl = (position["entry_price"] - exit_price) * position["position_size"]
        
        # Update metrics
        self.total_exposure -= position["current_value"]
        self.total_risk_amount -= position["risk_amount"]
        
        # Update symbol-specific risk
        if symbol in self.risk_per_symbol:
            self.risk_per_symbol[symbol] -= position["risk_amount"]
            if self.risk_per_symbol[symbol] <= 0:
                del self.risk_per_symbol[symbol]
        
        # Add to history
        close_data = position.copy()
        close_data["exit_price"] = exit_price
        close_data["exit_time"] = datetime.now().isoformat()
        close_data["exit_reason"] = exit_reason
        close_data["realized_pnl"] = realized_pnl
        close_data["duration_seconds"] = (
            datetime.fromisoformat(close_data["exit_time"]) - 
            datetime.fromisoformat(close_data["entry_time"])
        ).total_seconds()
        
        self.risk_history.append(close_data)
        
        # Limit history size
        if len(self.risk_history) > 100:
            self.risk_history = self.risk_history[-100:]
        
        # Remove from active positions
        del self.active_positions[trade_id]
        
        # Save state
        self._save_state()
        
        logger.info(f"Closed position {trade_id} for {symbol}, realized PnL: {realized_pnl:.2f}")
        
        return {
            "success": True,
            "realized_pnl": realized_pnl,
            "exit_reason": exit_reason
        }
    
    def calculate_total_risk(self) -> float:
        """
        Calculate current total risk exposure as a percentage of account balance
        
        Returns:
            Total risk exposure (as decimal percentage)
        """
        if self.initial_balance is None or self.initial_balance == 0:
            return 0.0
            
        return self.total_risk_amount / self.initial_balance
    
    def get_risk_metrics(self) -> Dict:
        """
        Get current risk metrics
        
        Returns:
            Risk metrics dictionary
        """
        total_risk_pct = self.calculate_total_risk()
        
        return {
            "total_exposure": self.total_exposure,
            "total_risk_amount": self.total_risk_amount,
            "total_risk_percentage": total_risk_pct * 100,
            "active_positions_count": len(self.active_positions),
            "positions_by_symbol": self._count_positions_by_symbol(),
            "risk_per_symbol": self.risk_per_symbol,
            "risk_limit_percentage": self.account_risk_limit * 100,
            "risk_utilization": (total_risk_pct / self.account_risk_limit * 100) if self.account_risk_limit > 0 else 0,
        }
    
    def _calculate_correlation_adjustment(self, new_symbol: str, market_data: Dict[str, pd.DataFrame]) -> float:
        """
        Calculate correlation-based risk adjustment factor
        
        Args:
            new_symbol: Symbol for the new trade
            market_data: Dictionary of market data for various symbols
            
        Returns:
            Correlation adjustment factor (0.5-1.5)
        """
        # If we don't have data for the new symbol, no adjustment
        if new_symbol not in market_data:
            return 1.0
        
        # If we don't have active positions, no adjustment needed
        active_symbols = set()
        for position in self.active_positions.values():
            active_symbols.add(position["symbol"])
        
        if not active_symbols:
            return 1.0
        
        # Calculate correlations between the new symbol and active symbols
        correlations = []
        
        for symbol in active_symbols:
            if symbol in market_data:
                # Extract price data
                new_prices = market_data[new_symbol]["close"].values
                existing_prices = market_data[symbol]["close"].values
                
                # Calculate correlation if we have enough data
                if len(new_prices) > 5 and len(existing_prices) > 5:
                    try:
                        # Try to match the lengths
                        min_length = min(len(new_prices), len(existing_prices))
                        correlation = np.corrcoef(
                            new_prices[-min_length:], 
                            existing_prices[-min_length:]
                        )[0, 1]
                        
                        # Only use valid correlation values
                        if not np.isnan(correlation):
                            correlations.append(abs(correlation))
                    except:
                        pass
        
        # If no valid correlations, return default
        if not correlations:
            return 1.0
        
        # Calculate average correlation
        avg_correlation = sum(correlations) / len(correlations)
        
        # Apply adjustment factor (lower position size for high correlation)
        # Adjust between 0.5 (high correlation) and 1.5 (negative correlation)
        adjustment_factor = 1.5 - avg_correlation
        
        # Ensure it's within reasonable bounds
        return max(0.5, min(1.5, adjustment_factor))
    
    def _count_positions_by_symbol(self) -> Dict[str, int]:
        """
        Count number of positions per symbol
        
        Returns:
            Dictionary mapping symbols to position counts
        """
        counts = {}
        for position in self.active_positions.values():
            symbol = position["symbol"]
            if symbol not in counts:
                counts[symbol] = 0
            counts[symbol] += 1
        return counts
    
    def update_account_balance(self, account_info: dict) -> None:
        # ...existing code...
        pass

    def update_trade_history(self, trade_result: dict) -> None:
        # ...existing code for updating trade history...
        self.trade_history.append(trade_result)
        # Update win rate if sufficient data
        if len(self.trade_history) >= 10:
            wins = len([t for t in self.trade_history if t.get("pnl", 0) > 0])
            self.win_rate = wins / len(self.trade_history) * 100
        # ...existing code...
    
    def calculer_risque_trade(self, opportunity: dict) -> float:
        """
        Calcule le montant à risquer sur un trade en fonction du capital actuel,
        du risque maximum par trade, du critère de Kelly et du win rate.
        
        Args:
            opportunity: Dictionnaire décrivant l'opportunité de trading.
                         (Peut inclure des facteurs supplémentaires si besoin)
        
        Returns:
            Montant à risquer (en valeur absolue).
        """
        # Calcul de risque de base (pourcentage du capital)
        risque_base = self.capital * self.max_risk_per_trade   # Correction ici (on n'applique pas /100)
        p = self.win_rate / 100
        kelly = max(0, p - (1 - p))
        kelly = min(kelly, self.kelly_fraction)
        montant_risque = risque_base * kelly
        return montant_risque

    def get_risk_state(self) -> dict:
        """
        Retourne l'état actuel du risque incluant le capital, le win rate
        et le montant de risque calculé par trade.
        """
        return {
            "capital": self.capital,
            "win_rate": self.win_rate,
            "montant_risque_par_trade": self.calculer_risque_trade({})  # opportunity factors can be added later
        }
    
    def set_market_cooldown(self, reason: str, duration_minutes: Optional[int] = None) -> None:
        """
        Set the market in cooldown mode, restricting new trades
        
        Args:
            reason: Reason for cooldown
            duration_minutes: Custom duration in minutes (uses default if None)
        """
        self.market_cooldown["active"] = True
        self.market_cooldown["start_time"] = datetime.now()
        self.market_cooldown["reason"] = reason
        
        if duration_minutes is not None:
            self.market_cooldown["duration"] = timedelta(minutes=duration_minutes)
        
        logger.warning(
            f"MARKET COOLDOWN ACTIVATED for {self.market_cooldown['duration'].total_seconds()/60} minutes. "
            f"Reason: {reason}"
        )
    
    def detect_market_volatility(self, symbol: str, market_data: pd.DataFrame) -> Dict:
        """
        Detect abnormal market volatility that might require cooldown
        
        Args:
            symbol: Trading pair symbol
            market_data: Recent market data
            
        Returns:
            Dict with volatility assessment
        """
        try:
            # Ensure we have enough data
            if len(market_data) < 10:
                return {"volatility_level": "unknown", "should_cooldown": False}
                
            # Calculate recent volatility
            close_prices = market_data['close'].values
            
            # Calculate short term price changes (5 candles)
            short_term_change = abs(close_prices[-1] / close_prices[-5] - 1) * 100
            
            # Determine volatility level
            volatility_level = "normal"
            should_cooldown = False
            cooldown_reason = None
            
            if short_term_change > self.volatility_thresholds["extreme"]:
                volatility_level = "extreme"
                should_cooldown = True
                cooldown_reason = f"Extreme volatility detected: {short_term_change:.2f}% change in short period"
            elif short_term_change > self.volatility_thresholds["high"]:
                volatility_level = "high"
                
            # If we should enter cooldown, activate it
            if should_cooldown:
                self.set_market_cooldown(cooldown_reason)
                
            return {
                "volatility_level": volatility_level,
                "short_term_change_pct": short_term_change,
                "should_cooldown": should_cooldown,
                "cooldown_reason": cooldown_reason
            }
                
        except Exception as e:
            logger.error(f"Error in volatility detection for {symbol}: {str(e)}")
            return {"volatility_level": "error", "should_cooldown": False}