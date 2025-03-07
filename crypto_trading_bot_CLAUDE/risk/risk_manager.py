"""
Risk management system for controlling trading exposure and preventing large losses
"""
import os
import json
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Any
from datetime import datetime

from utils.logger import setup_logger
from config.config import DATA_DIR

logger = setup_logger("risk_manager")

class RiskManager:
    """
    Manages trading risk by controlling position sizes and monitoring exposure
    """
    def __init__(self, default_risk_per_trade: float = 0.02, 
               account_risk_limit: float = 0.10,
               max_trades_per_symbol: int = 1,
               use_correlation_adjustment: bool = True):
        """
        Initialize the risk manager
        
        Args:
            default_risk_per_trade: Default risk per trade (as decimal percentage of account)
            account_risk_limit: Maximum total risk exposure (as decimal percentage of account)
            max_trades_per_symbol: Maximum number of concurrent trades per symbol
            use_correlation_adjustment: Whether to adjust risk based on correlation between assets
        """
        self.default_risk_per_trade = default_risk_per_trade
        self.account_risk_limit = account_risk_limit
        self.max_trades_per_symbol = max_trades_per_symbol
        self.use_correlation_adjustment = use_correlation_adjustment
        
        # Current active positions
        self.active_positions = {}  # trade_id -> position details
        
        # Risk history
        self.risk_history = []
        
        # Initial account balance (will be set later)
        self.initial_balance = None
        
        # Metrics
        self.total_exposure = 0.0
        self.total_risk_amount = 0.0
        self.risk_per_symbol = {}
        
        # Data path for persistence
        self.data_dir = os.path.join(DATA_DIR, "risk_management")
        os.makedirs(self.data_dir, exist_ok=True)
        self.data_path = os.path.join(self.data_dir, "risk_state.json")
        
        # Load previous state if exists
        self._load_state()
    
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
        
        if self.use_correlation_adjustment and market_data and len(market_data) > 1:
            correlation_factor = self._calculate_correlation_adjustment(symbol, market_data)
            position_size *= correlation_factor
        
        # Final position sizing and approval
        return {
            "approved": True,
            "position_sizing": {
                "position_size": position_size,
                "position_value": position_size * entry_price,
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
    
    def _save_state(self) -> None:
        """Save risk manager state to disk"""
        state = {
            "active_positions": self.active_positions,
            "risk_history": self.risk_history,
            "initial_balance": self.initial_balance,
            "total_exposure": self.total_exposure,
            "total_risk_amount": self.total_risk_amount,
            "risk_per_symbol": self.risk_per_symbol,
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
            
            logger.info(f"Loaded risk manager state: {len(self.active_positions)} active positions")
        except Exception as e:
            logger.error(f"Error loading risk manager state: {e}")