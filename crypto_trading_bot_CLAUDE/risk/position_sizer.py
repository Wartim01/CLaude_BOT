"""
Module for calculating position sizes based on risk parameters
and market conditions to optimize risk/reward
"""
import math
import numpy as np
from typing import Dict, List, Optional, Union
from datetime import datetime

from utils.logger import setup_logger
from risk.market_risk_analyzer import MarketRiskAnalyzer

logger = setup_logger("position_sizer")

class PositionSizer:
    """
    Class for calculating optimal position sizes based on 
    account balance, risk parameters, and market conditions
    """
    def __init__(self, default_risk_per_trade: float = 0.02, 
               max_risk_per_trade: float = 0.05,
               min_position_size: float = 0.0,
               account_risk_limit: float = 0.15,
               market_risk_analyzer: Optional[MarketRiskAnalyzer] = None):
        """
        Initialize the position sizer
        
        Args:
            default_risk_per_trade: Default percentage of account to risk per trade (0.02 = 2%)
            max_risk_per_trade: Maximum percentage of account to risk on a single trade
            min_position_size: Minimum allowed position size (in base currency)
            account_risk_limit: Maximum total account risk allowed at one time (0.15 = 15%)
            market_risk_analyzer: Optional market risk analyzer instance
        """
        self.default_risk_per_trade = default_risk_per_trade
        self.max_risk_per_trade = max_risk_per_trade
        self.min_position_size = min_position_size
        self.account_risk_limit = account_risk_limit
        self.market_risk_analyzer = market_risk_analyzer
        
        # Risk adjustment factors for different market risk levels
        self.risk_level_factors = {
            "low": 1.2,          # Increase position size in low-risk environments
            "medium": 1.0,       # Standard position size
            "high": 0.5,         # Half position size in high-risk environments
            "extreme": 0.25      # Quarter position size in extreme-risk environments
        }
        
        # Keep track of active positions and their risk
        self.active_positions = {}
        self.total_account_risk = 0.0
        
        # Default conversion rates for quote currency to account currency
        self.currency_rates = {}
        
        # Performance statistics
        self.sizing_stats = {
            "total_positions": 0,
            "avg_position_size": 0.0,
            "avg_risk_per_trade": 0.0,
            "largest_position": 0.0,
            "smallest_position": float('inf'),
            "risk_adjustments": []
        }
    
    def calculate_position_size(self, account_balance: float, 
                             entry_price: float, 
                             stop_loss: float,
                             symbol: str,
                             risk_per_trade: Optional[float] = None,
                             market_data: Optional[Dict[str, Union[dict, pd.DataFrame]]] = None,
                             quote_currency: str = 'USDT') -> Dict:
        """
        Calculate optimal position size based on account balance and risk parameters
        
        Args:
            account_balance: Current account balance (in account currency)
            entry_price: Entry price for the position
            stop_loss: Stop loss price
            symbol: Trading pair symbol
            risk_per_trade: Override default risk percentage (optional)
            market_data: Market data for risk assessment (optional)
            quote_currency: Currency of the quoted price
            
        Returns:
            Dictionary with position size details
        """
        # Input validation
        if entry_price <= 0 or stop_loss <= 0:
            return {
                "error": "Invalid price values",
                "position_size": 0.0
            }
        
        if entry_price == stop_loss:
            return {
                "error": "Entry price cannot equal stop loss price",
                "position_size": 0.0
            }
            
        # Determine if this is a long or short position
        is_long = entry_price > stop_loss
        
        # Calculate risk per trade (% of account balance)
        if risk_per_trade is None:
            risk_per_trade = self.default_risk_per_trade
        
        # Cap at maximum allowed risk per trade
        risk_per_trade = min(risk_per_trade, self.max_risk_per_trade)
        
        # Adjust risk based on market conditions if market data provided
        risk_adjustment = self._calculate_risk_adjustment(symbol, market_data)
        adjusted_risk = risk_per_trade * risk_adjustment
        
        # Calculate risk amount (in account currency)
        risk_amount = account_balance * adjusted_risk
        
        # Calculate position size based on risk and stop distance
        price_distance = abs(entry_price - stop_loss)
        distance_percentage = price_distance / entry_price
        
        # Formula: position_value = risk_amount / distance_percentage
        position_value = risk_amount / distance_percentage
        
        # Convert to position size in base currency
        position_size = position_value / entry_price
        
        # Check if the position size is within limits
        if position_size < self.min_position_size and position_size > 0:
            position_size = self.min_position_size
        
        # Check if this would exceed overall account risk limit
        new_total_risk = self.total_account_risk + adjusted_risk
        if new_total_risk > self.account_risk_limit:
            # Scale back the position to fit within account risk limit
            available_risk = max(0, self.account_risk_limit - self.total_account_risk)
            if available_risk > 0:
                scale_factor = available_risk / adjusted_risk
                position_size *= scale_factor
                adjusted_risk *= scale_factor
                risk_amount *= scale_factor
                
                logger.warning(f"Position size reduced due to account risk limit: {scale_factor:.2f} scale factor applied")
            else:
                position_size = 0.0
                logger.warning("Position rejected due to account risk limit")
        
        # Update stats
        self._update_sizing_stats(position_size, adjusted_risk)
        
        return {
            "position_size": float(position_size),
            "position_value": float(position_value),
            "risk_percentage": float(adjusted_risk * 100),  # Convert to percentage
            "risk_amount": float(risk_amount),
            "initial_risk": float(risk_per_trade * 100),    # Original risk before adjustments
            "adjustment_factor": float(risk_adjustment),
            "direction": "LONG" if is_long else "SHORT",
            "max_loss": float(risk_amount),
            "account_currency_value": float(position_value)
        }
    
    def calculate_stop_loss(self, entry_price: float, 
                         position_size: float, 
                         account_balance: float,
                         max_risk: Optional[float] = None,
                         direction: str = "LONG") -> Dict:
        """
        Calculate the stop loss price for a given position size and risk
        
        Args:
            entry_price: Entry price for the position
            position_size: Size of position in base currency
            account_balance: Current account balance
            max_risk: Maximum risk percentage (optional)
            direction: Position direction ("LONG" or "SHORT")
            
        Returns:
            Dictionary with stop loss details
        """
        if max_risk is None:
            max_risk = self.default_risk_per_trade
        
        # Cap at maximum risk
        max_risk = min(max_risk, self.max_risk_per_trade)
        
        # Calculate risk amount
        risk_amount = account_balance * max_risk
        
        # Calculate the required price move for the risk amount
        price_move = risk_amount / position_size
        
        # Calculate stop loss price based on direction
        stop_loss_price = None
        if direction.upper() == "LONG":
            stop_loss_price = entry_price - price_move
        else:  # SHORT
            stop_loss_price = entry_price + price_move
        
        # Ensure stop loss price is positive
        stop_loss_price = max(0.00001, stop_loss_price)
        
        return {
            "stop_loss_price": float(stop_loss_price),
            "price_distance": float(abs(entry_price - stop_loss_price)),
            "distance_percentage": float(abs(entry_price - stop_loss_price) / entry_price * 100),
            "risk_amount": float(risk_amount),
            "risk_percentage": float(max_risk * 100)
        }
    
    def register_position(self, position_id: str, 
                       symbol: str,
                       position_size: float,
                       entry_price: float,
                       stop_loss: float,
                       risk_percentage: float) -> bool:
        """
        Register a new active position to track overall account risk
        
        Args:
            position_id: Unique identifier for the position
            symbol: Trading pair symbol
            position_size: Size of position in base currency
            entry_price: Entry price for the position
            stop_loss: Stop loss price
            risk_percentage: Risk as percentage (0.02 = 2%)
            
        Returns:
            Success of registration
        """
        if position_id in self.active_positions:
            logger.warning(f"Position {position_id} already registered")
            return False
        
        self.active_positions[position_id] = {
            "symbol": symbol,
            "position_size": position_size,
            "entry_price": entry_price,
            "stop_loss": stop_loss,
            "risk_percentage": risk_percentage,
            "timestamp": datetime.now().isoformat()
        }
        
        # Update total account risk
        self.total_account_risk += risk_percentage
        
        logger.info(f"Position {position_id} registered. Total account risk: {self.total_account_risk:.2%}")
        return True
    
    def remove_position(self, position_id: str) -> bool:
        """
        Remove a position from active tracking when closed
        
        Args:
            position_id: Unique identifier for the position
            
        Returns:
            Success of removal
        """
        if position_id not in self.active_positions:
            logger.warning(f"Position {position_id} not found")
            return False
        
        # Reduce total account risk
        position = self.active_positions[position_id]
        self.total_account_risk -= position.get("risk_percentage", 0)
        self.total_account_risk = max(0, self.total_account_risk)  # Ensure never negative
        
        # Remove from active positions
        del self.active_positions[position_id]
        
        logger.info(f"Position {position_id} removed. Total account risk: {self.total_account_risk:.2%}")
        return True
    
    def get_available_risk_capacity(self) -> Dict:
        """
        Get the current available risk capacity
        
        Returns:
            Dictionary with risk capacity information
        """
        available_percentage = max(0, self.account_risk_limit - self.total_account_risk)
        
        return {
            "total_risk_limit": float(self.account_risk_limit * 100),  # As percentage
            "current_risk_usage": float(self.total_account_risk * 100),  # As percentage
            "available_risk": float(available_percentage * 100),  # As percentage
            "active_positions": len(self.active_positions),
            "capacity_used_percentage": float(self.total_account_risk / self.account_risk_limit * 100) if self.account_risk_limit > 0 else 0.0
        }
    
    def _calculate_risk_adjustment(self, symbol: str, 
                               market_data: Optional[Dict[str, Union[dict, pd.DataFrame]]] = None) -> float:
        """
        Calculate risk adjustment factor based on market conditions
        
        Args:
            symbol: Trading pair symbol
            market_data: Market data for risk assessment
            
        Returns:
            Risk adjustment factor (1.0 = no adjustment)
        """
        # Default to no adjustment
        adjustment_factor = 1.0
        adjustment_reason = "default"
        
        # If we have a market risk analyzer and market data, use it
        if self.market_risk_analyzer and market_data:
            try:
                # Get market risk assessment
                risk_assessment = self.market_risk_analyzer.calculate_market_risk(market_data)
                
                # Get risk level and apply corresponding factor
                risk_level = risk_assessment.get("level", "medium")
                adjustment_factor = self.risk_level_factors.get(risk_level, 1.0)
                adjustment_reason = f"market_risk_{risk_level}"
                
                # Fine-tune based on volatility
                volatility_factor = risk_assessment.get("risk_factors", {}).get("volatility", {})
                if "volatility_surge" in volatility_factor.get("factors", []):
                    # Further reduce position size in case of volatility surge
                    adjustment_factor *= 0.8
                    adjustment_reason += "_volatility_surge"
                elif "extreme_volatility" in volatility_factor.get("factors", []):
                    # Further reduce position size in case of extreme volatility
                    adjustment_factor *= 0.7
                    adjustment_reason += "_extreme_volatility"
                
                # Add to risk adjustment history for analysis
                self.sizing_stats["risk_adjustments"].append({
                    "symbol": symbol,
                    "factor": adjustment_factor,
                    "reason": adjustment_reason,
                    "risk_level": risk_level,
                    "timestamp": datetime.now().isoformat()
                })
                
                # Keep history size manageable
                if len(self.sizing_stats["risk_adjustments"]) > 100:
                    self.sizing_stats["risk_adjustments"] = self.sizing_stats["risk_adjustments"][-100:]
                
            except Exception as e:
                logger.error(f"Error calculating risk adjustment: {str(e)}")
        
        return adjustment_factor
    
    def _update_sizing_stats(self, position_size: float, risk_percentage: float) -> None:
        """
        Update position sizing statistics
        
        Args:
            position_size: Size of the position
            risk_percentage: Risk percentage for the position
        """
        self.sizing_stats["total_positions"] += 1
        
        # Update average position size and risk
        prev_total = self.sizing_stats["avg_position_size"] * (self.sizing_stats["total_positions"] - 1)
        self.sizing_stats["avg_position_size"] = (prev_total + position_size) / self.sizing_stats["total_positions"]
        
        prev_risk_total = self.sizing_stats["avg_risk_per_trade"] * (self.sizing_stats["total_positions"] - 1)
        self.sizing_stats["avg_risk_per_trade"] = (prev_risk_total + risk_percentage) / self.sizing_stats["total_positions"]
        
        # Update min/max
        if position_size > 0:
            self.sizing_stats["largest_position"] = max(self.sizing_stats["largest_position"], position_size)
            self.sizing_stats["smallest_position"] = min(self.sizing_stats["smallest_position"], position_size)
    
    def get_position_sizing_stats(self) -> Dict:
        """
        Get statistics about position sizing
        
        Returns:
            Dictionary with position sizing statistics
        """
        return {
            "total_positions": self.sizing_stats["total_positions"],
            "avg_position_size": float(self.sizing_stats["avg_position_size"]),
            "avg_risk_per_trade": float(self.sizing_stats["avg_risk_per_trade"] * 100),  # As percentage
            "largest_position": float(self.sizing_stats["largest_position"]),
            "smallest_position": float(self.sizing_stats["smallest_position"]) if self.sizing_stats["smallest_position"] < float('inf') else 0.0,
            "active_positions_count": len(self.active_positions),
            "total_account_risk": float(self.total_account_risk * 100),  # As percentage
            "recent_adjustments": self.sizing_stats["risk_adjustments"][-5:] if self.sizing_stats["risk_adjustments"] else []
        }
    
    def set_risk_level_factors(self, risk_factors: Dict[str, float]) -> None:
        """
        Update the risk level adjustment factors
        
        Args:
            risk_factors: Dictionary of risk level to adjustment factor
        """
        # Validate factors
        required_levels = ["low", "medium", "high", "extreme"]
        if all(level in risk_factors for level in required_levels):
            self.risk_level_factors = risk_factors
            logger.info(f"Risk level factors updated: {risk_factors}")
        else:
            logger.warning(f"Invalid risk factors provided, must include all levels: {required_levels}")