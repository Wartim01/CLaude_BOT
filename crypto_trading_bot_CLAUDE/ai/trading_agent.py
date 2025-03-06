"""
Trading agent that integrates AI predictions with trading logic
to make decisions and execute trades
"""
import os
import time
import json
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Union, Any
from datetime import datetime, timedelta
import threading
from enum import Enum

from config.config import DATA_DIR
from config.trading_params import (
    MAX_POSITIONS, RISK_PER_TRADE, DEFAULT_STOP_LOSS_PERCENT,
    TARGET_PROFIT_PERCENT, TIMEFRAMES
)
from utils.logger import setup_logger
from core.risk_manager import RiskManager
from core.position_tracker import PositionTracker
from ai.prediction_orchestrator import PredictionOrchestrator

logger = setup_logger("trading_agent")

class TradeState(Enum):
    """Enum for different trade states"""
    SEARCHING = "searching"  # Looking for opportunities
    MONITORING = "monitoring"  # Watching market for entry
    POSITIONING = "positioning"  # Preparing to enter
    ENTERED = "entered"  # In position
    MANAGING = "managing"  # Managing open position
    EXITING = "exiting"  # Exiting position
    STANDBY = "standby"  # Temporarily not trading

class TradingAgent:
    """
    Agent that integrates AI predictions with trading logic to
    make trading decisions and execute trades
    """
    def __init__(self, 
                 risk_manager: Optional[RiskManager] = None,
                 position_tracker: Optional[PositionTracker] = None,
                 prediction_orchestrator: Optional[PredictionOrchestrator] = None,
                 trading_enabled: bool = False):
        """
        Initialize the trading agent
        
        Args:
            risk_manager: Risk manager instance
            position_tracker: Position tracker instance
            prediction_orchestrator: Prediction orchestrator instance
            trading_enabled: Whether trading is enabled initially
        """
        # Core components
        self.risk_manager = risk_manager
        self.position_tracker = position_tracker
        self.prediction_orchestrator = prediction_orchestrator or PredictionOrchestrator()
        
        # Trading state
        self.trading_enabled = trading_enabled
        self.trading_state = TradeState.STANDBY if not trading_enabled else TradeState.SEARCHING
        self.watched_symbols = {}  # {symbol: {state, last_update, data}}
        
        # Configuration
        self.confidence_threshold = 0.65  # Min confidence to consider a trade
        self.min_strength = 6  # Minimum strength (0-10) for a trade
        self.max_positions = MAX_POSITIONS
        self.risk_per_trade = RISK_PER_TRADE
        self.retry_interval = 60  # seconds to wait after failed trade
        
        # Auto trading thread
        self.trading_thread = None
        self.should_stop = False
        self.trading_interval = 300  # seconds between trading cycles
        
        # Trading metrics
        self.trading_metrics = {
            "trades_executed": 0,
            "successful_trades": 0,
            "failed_trades": 0,
            "total_profit": 0.0,
            "total_loss": 0.0,
            "net_pnl": 0.0,
            "win_rate": 0.0
        }
        
        # Trade decisions history
        self.trade_decisions = []
        self.max_decisions_history = 1000
        
        # Trading log
        self.trades_dir = os.path.join(DATA_DIR, "trades")
        self.metrics_path = os.path.join(self.trades_dir, "trading_metrics.json")
        self.decisions_path = os.path.join(self.trades_dir, "trade_decisions.json")
        
        # Ensure directories exist
        os.makedirs(self.trades_dir, exist_ok=True)
        
        # Load trading metrics and history
        self._load_trading_data()
        
        logger.info("Trading agent initialized")
    
    def _load_trading_data(self):
        """Load trading metrics and history"""
        try:
            # Load trading metrics
            if os.path.exists(self.metrics_path):
                with open(self.metrics_path, 'r') as f:
                    self.trading_metrics = json.load(f)
                logger.info("Trading metrics loaded")
            
            # Load trade decisions history
            if os.path.exists(self.decisions_path):
                with open(self.decisions_path, 'r') as f:
                    self.trade_decisions = json.load(f)
                logger.info("Trade decisions history loaded")
        except Exception as e:
            logger.error(f"Error loading trading data: {str(e)}")
    
    def _save_trading_data(self):
        """Save trading metrics and history"""
        try:
            # Save trading metrics
            with open(self.metrics_path, 'w') as f:
                json.dump(self.trading_metrics, f, indent=2)
            
            # Limit the size of decisions history
            if len(self.trade_decisions) > self.max_decisions_history:
                self.trade_decisions = self.trade_decisions[-self.max_decisions_history:]
            
            # Save trade decisions history
            with open(self.decisions_path, 'w') as f:
                json.dump(self.trade_decisions, f, indent=2)
            
            logger.debug("Trading data saved")
        except Exception as e:
            logger.error(f"Error saving trading data: {str(e)}")
    
    def start_auto_trading(self):
        """
        Start automatic trading in background thread
        """
        if not self.trading_enabled:
            logger.warning("Cannot start auto trading - trading is disabled")
            return False
        
        if self.trading_thread is None or not self.trading_thread.is_alive():
            self.should_stop = False
            self.trading_thread = threading.Thread(target=self._trading_worker)
            self.trading_thread.daemon = True
            self.trading_thread.start()
            
            self.trading_state = TradeState.SEARCHING
            logger.info(f"Auto trading started (interval: {self.trading_interval}s)")
            return True
        
        return False
    
    def stop_auto_trading(self):
        """
        Stop automatic trading
        """
        self.should_stop = True
        if self.trading_thread and self.trading_thread.is_alive():
            self.trading_thread.join(timeout=2.0)
        
        self.trading_state = TradeState.STANDBY
        logger.info("Auto trading stopped")
        return True
    
    def _trading_worker(self):
        """
        Background worker thread for automatic trading
        """
        while not self.should_stop:
            try:
                # Main trading cycle
                self._trading_cycle()
                
                # Sleep until next cycle
                time.sleep(self.trading_interval)
            except Exception as e:
                logger.error(f"Error in trading worker: {str(e)}")
                time.sleep(self.retry_interval)
    
    def _trading_cycle(self):
        """
        One complete trading cycle:
        1. Check existing positions
        2. Look for new opportunities
        3. Execute trades if conditions are met
        """
        # Skip if trading is disabled
        if not self.trading_enabled:
            return
        
        # 1. Check and manage existing positions
        if self.position_tracker:
            self._manage_positions()
        
        # 2. Look for new trading opportunities (if we have capacity)
        if self._has_position_capacity():
            self._find_opportunities()
    
    def _manage_positions(self):
        """
        Check and manage all open positions
        """
        if not self.position_tracker:
            logger.warning("Cannot manage positions - position tracker not available")
            return
        
        try:
            # Get all open positions
            open_positions = self.position_tracker.get_all_open_positions()
            
            for symbol, positions in open_positions.items():
                for position in positions:
                    # Set state to managing for this symbol
                    if symbol in self.watched_symbols:
                        self.watched_symbols[symbol]["state"] = TradeState.MANAGING
                    
                    # Check if we should modify or close the position
                    self._check_position_exit(symbol, position)
        except Exception as e:
            logger.error(f"Error managing positions: {str(e)}")
    
    def _check_position_exit(self, symbol: str, position: Dict):
        """
        Check if a position should be exited or modified
        
        Args:
            symbol: Trading symbol
            position: Position data dictionary
        """
        if not self.position_tracker or not self.risk_manager:
            return
        
        try:
            # Get latest market data
            market_data = self.position_tracker.get_symbol_data(symbol)
            if market_data is None or market_data.empty:
                logger.warning(f"No market data available for {symbol}")
                return
            
            # Check if this is a trade we're tracking
            position_id = position.get("id")
            is_tracked = position.get("metadata", {}).get("managed_by_agent", False)
            
            if not is_tracked:
                # Skip positions not managed by this agent
                logger.debug(f"Skipping unmanaged position: {position_id}")
                return
            
            # Get current price
            current_price = float(market_data['close'].iloc[-1])
            
            # Get position details
            entry_price = position.get("entry_price", 0)
            stop_loss = position.get("stop_loss", 0)
            take_profit = position.get("take_profit", 0)
            position_type = position.get("type", "UNKNOWN")
            
            # Calculate unrealized PnL
            if position_type == "LONG":
                unrealized_pnl_pct = (current_price - entry_price) / entry_price * 100
            else:  # SHORT
                unrealized_pnl_pct = (entry_price - current_price) / entry_price * 100
            
            # Check if stop loss or take profit has been hit
            should_close = False
            close_reason = ""
            
            if position_type == "LONG":
                if stop_loss > 0 and current_price <= stop_loss:
                    should_close = True
                    close_reason = "stop_loss"
                elif take_profit > 0 and current_price >= take_profit:
                    should_close = True
                    close_reason = "take_profit"
            else:  # SHORT
                if stop_loss > 0 and current_price >= stop_loss:
                    should_close = True
                    close_reason = "stop_loss"
                elif take_profit > 0 and current_price <= take_profit:
                    should_close = True
                    close_reason = "take_profit"
            
            # Get new prediction to see if we should exit based on signal change
            prediction_result = self.prediction_orchestrator.get_prediction(
                symbol, market_data, include_details=False
            )
            
            if prediction_result["success"]:
                prediction = prediction_result["prediction"]
                
                # Check if prediction now contradicts position direction
                if position_type == "LONG" and prediction["direction"] == "SELL":
                    should_close = True
                    close_reason = "signal_change"
                elif position_type == "SHORT" and prediction["direction"] == "BUY":
                    should_close = True
                    close_reason = "signal_change"
            
            # Execute position closure if needed
            if should_close:
                close_result = self.position_tracker.close_position(
                    symbol=symbol,
                    position_id=position_id,
                    current_price=current_price
                )
                
                if close_result.get("success", False):
                    # Record trade result
                    self._record_trade_result(symbol, position, close_result, close_reason)
                    
                    logger.info(f"Closed position {position_id} for {symbol} ({close_reason}) - PnL: {unrealized_pnl_pct:.2f}%")
                else:
                    logger.error(f"Failed to close position {position_id}: {close_result.get('error', 'Unknown error')}")
            
            # If not closing, see if we should adjust stop loss (trailing)
            elif position.get("metadata", {}).get("use_trailing_stop", True):
                self._adjust_trailing_stop(symbol, position, current_price)
            
        except Exception as e:
            logger.error(f"Error checking position exit for {symbol}: {str(e)}")
    
    def _adjust_trailing_stop(self, symbol: str, position: Dict, current_price: float):
        """
        Adjust trailing stop loss for a position
        
        Args:
            symbol: Trading symbol
            position: Position data
            current_price: Current market price
        """
        if not self.position_tracker:
            return
        
        try:
            # Get position details
            position_id = position.get("id")
            position_type = position.get("type", "UNKNOWN")
            current_stop = position.get("stop_loss", 0)
            
            if current_stop <= 0:
                return  # No stop loss set
            
            # Calculate potential new stop
            trail_pct = position.get("metadata", {}).get("trailing_stop_pct", 2.0)
            trail_activation_pct = position.get("metadata", {}).get("trailing_activation_pct", 1.0)
            
            if position_type == "LONG":
                # For long positions, trail upwards only
                trail_distance = current_price * (trail_pct / 100)
                new_stop = current_price - trail_distance
                
                # Only move stop loss up, never down
                if new_stop > current_stop:
                    # Update stop loss
                    update_result = self.position_tracker.update_position_stop_loss(
                        symbol=symbol,
                        position_id=position_id,
                        new_stop_loss=new_stop
                    )
                    
                    if update_result.get("success", False):
                        logger.info(f"Updated trailing stop for {symbol} position {position_id}: {new_stop:.2f}")
            elif position_type == "SHORT":
                # For short positions, trail downwards only
                trail_distance = current_price * (trail_pct / 100)
                new_stop = current_price + trail_distance
                
                # Only move stop loss down, never up
                if new_stop < current_stop:
                    # Update stop loss
                    update_result = self.position_tracker.update_position_stop_loss(
                        symbol=symbol,
                        position_id=position_id,
                        new_stop_loss=new_stop
                    )
                    
                    if update_result.get("success", False):
                        logger.info(f"Updated trailing stop for {symbol} position {position_id}: {new_stop:.2f}")
        except Exception as e:
            logger.error(f"Error adjusting trailing stop for {symbol}: {str(e)}")
    
    def _has_position_capacity(self) -> bool:
        """
        Check if we have capacity for new positions
        
        Returns:
            True if we have capacity, False otherwise
        """
        if not self.position_tracker:
            return False
        
        try:
            # Count current open positions
            open_positions = self.position_tracker.get_all_open_positions()
            total_positions = sum(len(pos) for pos in open_positions.values())
            
            # Check if we're under the maximum
            return total_positions < self.max_positions
        except Exception as e:
            logger.error(f"Error checking position capacity: {str(e)}")
            return False
    
    def _find_opportunities(self):
        """
        Look for new trading opportunities
        """
        # Get symbols to check
        symbols = self._get_tradable_symbols()
        
        for symbol in symbols:
            try:
                # Skip symbols we're already watching or trading
                if symbol in self.watched_symbols and self.watched_symbols[symbol]["state"] not in [
                    TradeState.SEARCHING, TradeState.STANDBY
                ]:
                    continue
                
                # Get market data
                market_data = self._get_symbol_data(symbol)
                if market_data is None or market_data.empty:
                    continue
                
                # Get prediction
                prediction_result = self.prediction_orchestrator.get_prediction(
                    symbol, market_data, include_details=True
                )
                
                if not prediction_result["success"]:
                    continue
                
                # Evaluate prediction for trading
                trade_decision = self._evaluate_prediction(symbol, prediction_result, market_data)
                
                # Record the decision
                self._record_trade_decision(symbol, prediction_result, trade_decision)
                
                # If we should trade, execute it
                if trade_decision["should_trade"]:
                    self._execute_trade(symbol, trade_decision)
            
            except Exception as e:
                logger.error(f"Error finding opportunity for {symbol}: {str(e)}")
    
    def _get_tradable_symbols(self) -> List[str]:
        """
        Get list of symbols to check for trading opportunities
        
        Returns:
            List of tradable symbols
        """
        if self.position_tracker:
            # Get list of available symbols from position tracker
            available_symbols = self.position_tracker.get_available_symbols()
            
            # Filter to symbols with minimum criteria
            tradable = []
            for symbol in available_symbols:
                # Check if symbol meets basic requirements
                # For now, we'll accept all available symbols
                tradable.append(symbol)
            
            return tradable
        
        # Default test symbols
        return ["BTCUSDT", "ETHUSDT"]
    
    def _get_symbol_data(self, symbol: str) -> pd.DataFrame:
        """
        Get market data for a symbol
        
        Args:
            symbol: Trading symbol
        
        Returns:
            DataFrame with OHLCV data
        """
        if self.position_tracker:
            return self.position_tracker.get_symbol_data(symbol)
        
        return None
    
    def _evaluate_prediction(self, symbol: str, prediction_result: Dict, market_data: pd.DataFrame) -> Dict:
        """
        Evaluate a prediction to decide if we should trade
        
        Args:
            symbol: Trading symbol
            prediction_result: Result from prediction_orchestrator
            market_data: Market data DataFrame
        
        Returns:
            Trade decision dictionary
        """
        prediction = prediction_result["prediction"]
        
        # Initialize decision
        decision = {
            "should_trade": False,
            "direction": "NEUTRAL",
            "entry_price": 0,
            "stop_loss": 0,
            "take_profit": 0,
            "confidence": prediction["confidence"],
            "strength": prediction["strength"],
            "signals": prediction.get("signals", []),
            "reason": "neutral_signal"
        }
        
        # Check basic signal
        if prediction["direction"] not in ["BUY", "SELL"]:
            return decision
        
        # Check confidence and strength
        if prediction["confidence"] < self.confidence_threshold or prediction["strength"] < self.min_strength:
            decision["reason"] = "below_threshold"
            return decision
        
        # Check if the risk manager approves this trade
        risk_check = True  # Default to True
        if self.risk_manager:
            risk_check = self.risk_manager.check_trade_risk(
                symbol=symbol,
                trade_direction=prediction["direction"],
                confidence=prediction["confidence"]
            )
        
        if not risk_check:
            decision["reason"] = "risk_manager_rejected"
            return decision
        
        # Calculate entry, stop loss, and take profit
        current_price = float(market_data['close'].iloc[-1])
        direction = prediction["direction"]
        
        # For simplicity, using fixed percentages for SL/TP
        # In a real implementation, these would be adaptive based on volatility
        stop_loss_pct = DEFAULT_STOP_LOSS_PERCENT
        take_profit_pct = TARGET_PROFIT_PERCENT
        
        # Set decision parameters
        decision["should_trade"] = True
        decision["direction"] = direction
        decision["entry_price"] = current_price
        
        if direction == "BUY":
            decision["stop_loss"] = current_price * (1 - stop_loss_pct/100)
            decision["take_profit"] = current_price * (1 + take_profit_pct/100)
        else:  # SELL
            decision["stop_loss"] = current_price * (1 + stop_loss_pct/100)
            decision["take_profit"] = current_price * (1 - take_profit_pct/100)
        
        return decision
    
    def _execute_trade(self, symbol: str, decision: Dict) -> Dict:
        """
        Execute a trade based on decision
        
        Args:
            symbol: Trading symbol
            decision: Trade decision dictionary
        
        Returns:
            Trade execution results
        """
        if not self.position_tracker:
            logger.warning(f"Cannot execute trade for {symbol} - position tracker not available")
            return {"success": False, "error": "Position tracker not available"}
        
        if not decision["should_trade"]:
            return {"success": False, "error": "Decision indicates not to trade"}
        
        try:
            # Calculate position size
            position_size = self._calculate_position_size(symbol, decision)
            
            # Prepare trade metadata
            metadata = {
                "managed_by_agent": True,
                "use_trailing_stop": True,
                "trailing_stop_pct": 2.0,  # 2% trailing stop
                "trailing_activation_pct": 1.0,  # Activate after 1% profit
                "strategy": "ai_prediction",
                "confidence": decision["confidence"],
                "strength": decision["strength"],
                "signals": decision["signals"],
                "prediction_id": datetime.now().isoformat()
            }
            
            # Execute the trade
            position_result = self.position_tracker.open_position(
                symbol=symbol,
                position_type=decision["direction"],
                entry_price=decision["entry_price"],
                stop_loss=decision["stop_loss"],
                take_profit=decision["take_profit"],
                quantity=position_size,
                metadata=metadata
            )
            
            if position_result.get("success", False):
                logger.info(f"Executed trade for {symbol}: {decision['direction']} {position_size:.6f} @ {decision['entry_price']:.2f}")
                
                # Update watched symbols
                self.watched_symbols[symbol] = {
                    "state": TradeState.ENTERED,
                    "last_update": datetime.now().isoformat(),
                    "position_id": position_result.get("position_id"),
                    "direction": decision["direction"]
                }
                
                # Update trading metrics
                self.trading_metrics["trades_executed"] += 1
                
                # Save trading data
                self._save_trading_data()
                
                return {
                    "success": True,
                    "position_id": position_result.get("position_id"),
                    "entry_price": decision["entry_price"],
                    "direction": decision["direction"]
                }
            else:
                logger.error(f"Failed to execute trade for {symbol}: {position_result.get('error', 'Unknown error')}")
                return {"success": False, "error": position_result.get("error", "Unknown error")}
        
        except Exception as e:
            logger.error(f"Error executing trade for {symbol}: {str(e)}")
            return {"success": False, "error": str(e)}
    
    def _calculate_position_size(self, symbol: str, decision: Dict) -> float:
        """
        Calculate position size based on risk parameters
        
        Args:
            symbol: Trading symbol
            decision: Trade decision
            
        Returns:
            Position size in units of base asset
        """
        if not self.position_tracker:
            return 0.0
        
        try:
            # Get account balance
            balance = self.position_tracker.get_account_balance()
            
            if not balance or balance <= 0:
                return 0.0
            
            # Calculate risk amount
            risk_amount = balance * self.risk_per_trade
            
            # Calculate stop loss distance in percentage
            entry = decision["entry_price"]
            stop = decision["stop_loss"]
            
            if entry <= 0 or stop <= 0:
                return 0.0
            
            if decision["direction"] == "BUY":
                stop_distance_pct = (entry - stop) / entry
            else:  # SELL
                stop_distance_pct = (stop - entry) / entry
            
            if stop_distance_pct <= 0:
                logger.warning(f"Invalid stop distance for {symbol}: {stop_distance_pct}")
                return 0.0
            
            # Calculate position size based on risk and stop distance
            position_value = risk_amount / stop_distance_pct
            
            # Limit position size to a percentage of account
            max_position_value = balance * 0.2  # Max 20% of account per position
            position_value = min(position_value, max_position_value)
            
            # Calculate quantity
            quantity = position_value / entry
            
            # Apply symbol-specific rounding
            # This would normally come from exchange info
            quantity = round(quantity, 6)
            
            return quantity
            
        except Exception as e:
            logger.error(f"Error calculating position size for {symbol}: {str(e)}")
            return 0.0
    
    def _record_trade_decision(self, symbol: str, prediction_result: Dict, decision: Dict):
        """
        Record a trade decision for analysis
        
        Args:
            symbol: Trading symbol
            prediction_result: Result from prediction_orchestrator
            decision: Trade decision
        """
        entry = {
            "timestamp": datetime.now().isoformat(),
            "symbol": symbol,
            "prediction": prediction_result["prediction"],
            "decision": decision,
            "models": prediction_result.get("model_predictions", {}),
            "executed": decision["should_trade"]
        }
        
        # Add to decisions history
        self.trade_decisions.append(entry)
        
        # Save trading data
        self._save_trading_data()
    
    def _record_trade_result(self, symbol: str, position: Dict, close_result: Dict, reason: str):
        """
        Record the result of a closed trade
        
        Args:
            symbol: Trading symbol
            position: Position data
            close_result: Result from position closure
            reason: Reason for closure
        """
        # Calculate PnL
        entry_price = position.get("entry_price", 0)
        exit_price = close_result.get("exit_price", 0)
        position_type = position.get("type", "UNKNOWN")
        
        if position_type == "LONG":
            pnl_pct = (exit_price - entry_price) / entry_price * 100 if entry_price > 0 else 0
        else:  # SHORT
            pnl_pct = (entry_price - exit_price) / entry_price * 100 if entry_price > 0 else 0
        
        # Get position size and calculate absolute PnL
        quantity = position.get("quantity", 0)
        absolute_pnl = (exit_price - entry_price) * quantity if position_type == "LONG" else (entry_price - exit_price) * quantity
        
        # Update trading metrics
        self.trading_metrics["trades_executed"] += 1
        
        if pnl_pct > 0:
            self.trading_metrics["successful_trades"] += 1
            self.trading_metrics["total_profit"] += absolute_pnl
        else:
            self.trading_metrics["failed_trades"] += 1
            self.trading_metrics["total_loss"] += absolute_pnl
        
        # Calculate total PnL and win rate
        self.trading_metrics["net_pnl"] = self.trading_metrics["total_profit"] + self.trading_metrics["total_loss"]
        
        if self.trading_metrics["trades_executed"] > 0:
            self.trading_metrics["win_rate"] = self.trading_metrics["successful_trades"] / self.trading_metrics["trades_executed"] * 100
        
        # Save trading data
        self._save_trading_data()
        
        # Update prediction model performance
        if "prediction_id" in position.get("metadata", {}):
            # Extract prediction ID used for this trade
            prediction_id = position["metadata"]["prediction_id"]
            
            # Determine actual outcome
            if pnl_pct > 0:
                actual_outcome = "BULLISH" if position_type == "LONG" else "BEARISH"
            else:
                actual_outcome = "BEARISH" if position_type == "LONG" else "BULLISH"
            
            # Update model performance metrics
            try:
                self.prediction_orchestrator.update_model_performance(
                    symbol=symbol,
                    prediction_id=prediction_id,
                    actual_outcome=actual_outcome,
                    pnl=absolute_pnl
                )
            except Exception as e:
                logger.error(f"Error updating model performance metrics: {str(e)}")
    
    def get_status(self) -> Dict:
        """
        Get current status of the trading agent
        
        Returns:
            Status dictionary
        """
        # Count current positions
        open_positions = 0
        active_symbols = []
        
        if self.position_tracker:
            positions = self.position_tracker.get_all_open_positions()
            open_positions = sum(len(pos) for pos in positions.values())
            active_symbols = list(positions.keys())
        
        # Get overall statistics
        return {
            "trading_enabled": self.trading_enabled,
            "trading_state": self.trading_state.value,
            "open_positions": open_positions,
            "active_symbols": active_symbols,
            "watched_symbols": list(self.watched_symbols.keys()),
            "trades_executed": self.trading_metrics["trades_executed"],
            "successful_trades": self.trading_metrics["successful_trades"],
            "win_rate": f"{self.trading_metrics['win_rate']:.2f}%",
            "net_pnl": self.trading_metrics["net_pnl"],
            "auto_trading": self.trading_thread is not None and self.trading_thread.is_alive()
        }
    
    def enable_trading(self, enabled: bool = True) -> bool:
        """
        Enable or disable trading
        
        Args:
            enabled: Whether to enable trading
            
        Returns:
            Success of the operation
        """
        self.trading_enabled = enabled
        
        if enabled:
            self.trading_state = TradeState.SEARCHING
            logger.info("Trading enabled")
        else:
            self.trading_state = TradeState.STANDBY
            # Stop auto trading if running
            if self.trading_thread and self.trading_thread.is_alive():
                self.stop_auto_trading()
            logger.info("Trading disabled")
        
        return True
    
    def manual_trade(self, symbol: str, direction: str) -> Dict:
        """
        Execute a manual trade based on specified direction
        
        Args:
            symbol: Trading symbol
            direction: Trade direction ('BUY' or 'SELL')
            
        Returns:
            Result of the trade execution
        """
        if not self.trading_enabled:
            return {"success": False, "error": "Trading is disabled"}
        
        if direction not in ["BUY", "SELL"]:
            return {"success": False, "error": "Invalid trade direction"}
        
        if not self.position_tracker:
            return {"success": False, "error": "Position tracker not available"}
        
        try:
            # Get market data
            market_data = self._get_symbol_data(symbol)
            if market_data is None or market_data.empty:
                return {"success": False, "error": "No market data available"}
            
            # Create a simple decision
            current_price = float(market_data['close'].iloc[-1])
            
            decision = {
                "should_trade": True,
                "direction": direction,
                "entry_price": current_price,
                "confidence": 0.9,  # High confidence for manual trades
                "strength": 10,     # Max strength for manual trades
                "signals": ["manual_trade"]
            }
            
            # Add stop loss and take profit
            if direction == "BUY":
                decision["stop_loss"] = current_price * (1 - DEFAULT_STOP_LOSS_PERCENT/100)
                decision["take_profit"] = current_price * (1 + TARGET_PROFIT_PERCENT/100)
            else:  # SELL
                decision["stop_loss"] = current_price * (1 + DEFAULT_STOP_LOSS_PERCENT/100)
                decision["take_profit"] = current_price * (1 - TARGET_PROFIT_PERCENT/100)
            
            # Execute the trade
            result = self._execute_trade(symbol, decision)
            
            return result
            
        except Exception as e:
            logger.error(f"Error executing manual trade for {symbol}: {str(e)}")
            return {"success": False, "error": str(e)}