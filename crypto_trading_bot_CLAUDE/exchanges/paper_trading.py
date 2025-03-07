"""
Paper trading client for simulation
"""
import os
import json
import time
import uuid
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta

from utils.logger import setup_logger
from exchanges.exchange_client import ExchangeClient
from config.config import DATA_DIR

logger = setup_logger("paper_trading")

class PaperTradingClient(ExchangeClient):
    """
    Paper trading client for simulating trades without real money
    """
    def __init__(self, starting_balance: float = 10000, quote_currency: str = "USDT",
                external_data_feed: Any = None, price_slippage: float = 0.0005,
                data_source: Optional[str] = None):
        """
        Initialize paper trading client
        
        Args:
            starting_balance: Initial balance
            quote_currency: Quote currency symbol
            external_data_feed: Optional external data feed for prices
            price_slippage: Simulated price slippage percentage
            data_source: Optional data source for historical prices
        """
        self.quote_currency = quote_currency
        self.starting_balance = starting_balance
        self.balance = {
            "total": starting_balance,
            "free": starting_balance,
            "used": 0.0
        }
        
        # External data feed for live prices
        self.external_data_feed = external_data_feed
        
        # Simulated price slippage (as a percentage)
        self.price_slippage = price_slippage
        
        # Order and position tracking
        self.orders = {}  # order_id -> order_details
        self.positions = {}  # symbol -> position_details
        self.order_history = []
        self.executed_trades = []
        
        # Data source for historical prices
        self.data_source = data_source
        self.price_cache = {}  # symbol -> latest price
        
        # Local storage
        self.storage_dir = os.path.join(DATA_DIR, "paper_trading")
        os.makedirs(self.storage_dir, exist_ok=True)
        
        # Load state if it exists
        self.load_state()
        
        logger.info(f"Paper trading client initialized with {starting_balance} {quote_currency}")
    
    def test_connection(self) -> bool:
        """Always returns True for paper trading"""
        return True
    
    def get_balance(self) -> Dict:
        """Get current balance"""
        # Update total balance based on positions
        total_position_value = 0.0
        
        for symbol, position in self.positions.items():
            current_price = self._get_current_price(symbol)
            position_value = position["amount"] * current_price
            total_position_value += position_value
        
        self.balance["total"] = self.balance["free"] + self.balance["used"] + total_position_value
        
        return self.balance
    
    def fetch_ohlcv(self, symbol: str, timeframe: str, limit: int = 100) -> List:
        """
        Fetch OHLCV data (delegates to external data feed if available)
        """
        if self.external_data_feed and hasattr(self.external_data_feed, 'fetch_ohlcv'):
            return self.external_data_feed.fetch_ohlcv(symbol, timeframe, limit)
        
        # Fallback to mock data if no external feed
        now = datetime.now()
        mock_data = []
        
        # Convert timeframe to minutes
        minutes = self._timeframe_to_minutes(timeframe)
        
        current_price = self._get_current_price(symbol)
        
        for i in range(limit):
            timestamp = int((now - timedelta(minutes=minutes * (limit - i - 1))).timestamp() * 1000)
            
            # Generate random OHLCV data around current price
            close = current_price * (1 + 0.0001 * np.random.randn())
            
            # Basic simulation of a candle
            high = close * (1 + abs(0.0005 * np.random.randn()))
            low = close * (1 - abs(0.0005 * np.random.randn()))
            open_price = close * (1 + 0.0005 * np.random.randn())
            volume = abs(1000 * np.random.randn() + 10000)
            
            mock_data.append([timestamp, open_price, high, low, close, volume])
        
        return mock_data
    
    def create_order(self, symbol: str, order_type: str, side: str, amount: float, price: float = None) -> Dict:
        """
        Create a paper trading order
        
        Args:
            symbol: Symbol to trade
            order_type: Order type (market, limit, etc.)
            side: Order side (buy or sell)
            amount: Order amount
            price: Order price (for limit orders)
            
        Returns:
            Order information
        """
        # Validate inputs
        if amount <= 0:
            return {"success": False, "error": "Invalid amount"}
        
        # Get current price
        current_price = self._get_current_price(symbol)
        
        # For limit orders, use the specified price
        # For market orders, simulate small slippage
        if order_type.lower() == "market":
            if side.lower() == "buy":
                execution_price = current_price * (1 + self.price_slippage)
            else:
                execution_price = current_price * (1 - self.price_slippage)
        else:
            # For limit orders, use the specified price
            if price is None:
                return {"success": False, "error": "Price is required for limit orders"}
            execution_price = price
        
        # Generate order ID
        order_id = f"paper_{int(time.time())}_{uuid.uuid4().hex[:6]}"
        
        # Calculate order value
        order_value = amount * execution_price
        
        # Check if we have enough balance for BUY orders
        if side.lower() == "buy":
            if self.balance["free"] < order_value:
                return {"success": False, "error": "Insufficient balance"}
            
            # Update balance
            self.balance["free"] -= order_value
            self.balance["used"] += order_value
        
        # Create order
        order = {
            "id": order_id,
            "symbol": symbol,
            "type": order_type.lower(),
            "side": side.lower(),
            "amount": amount,
            "price": execution_price,
            "value": order_value,
            "status": "open",
            "created_at": datetime.now().isoformat(),
            "updated_at": datetime.now().isoformat()
        }
        
        self.orders[order_id] = order
        
        # For market orders, execute immediately
        if order_type.lower() == "market":
            self._execute_order(order_id)
        
        # Save state
        self.save_state()
        
        return {
            "success": True,
            "order_id": order_id,
            "price": execution_price,
            "amount": amount,
            "side": side.lower()
        }
    
    def get_order(self, order_id: str, symbol: str = None) -> Dict:
        """Get order details"""
        if order_id in self.orders:
            return {"success": True, "order": self.orders[order_id]}
        
        # Check order history
        for order in self.order_history:
            if order["id"] == order_id:
                return {"success": True, "order": order}
        
        return {"success": False, "error": "Order not found"}
    
    def cancel_order(self, order_id: str, symbol: str = None) -> Dict:
        """Cancel an order"""
        if order_id not in self.orders:
            return {"success": False, "error": "Order not found"}
        
        order = self.orders[order_id]
        
        # Can only cancel open orders
        if order["status"] != "open":
            return {"success": False, "error": f"Order status is {order['status']}, cannot cancel"}
        
        # Update order status
        order["status"] = "canceled"
        order["updated_at"] = datetime.now().isoformat()
        
        # If it was a buy order, release the funds
        if order["side"] == "buy":
            self.balance["free"] += order["value"]
            self.balance["used"] -= order["value"]
        
        # Move to history
        self.order_history.append(order)
        del self.orders[order_id]
        
        # Save state
        self.save_state()
        
        return {"success": True, "order_id": order_id}
    
    def get_open_positions(self) -> List[Dict]:
        """Get open positions"""
        positions = []
        
        for symbol, position in self.positions.items():
            # Get current price to calculate unrealized PnL
            current_price = self._get_current_price(symbol)
            
            # Calculate unrealized PnL
            if position["side"] == "buy":
                pnl_percentage = (current_price - position["entry_price"]) / position["entry_price"]
            else:
                pnl_percentage = (position["entry_price"] - current_price) / position["entry_price"]
                
            pnl_amount = pnl_percentage * position["value"]
            
            # Add to results
            positions.append({
                "symbol": symbol,
                "side": position["side"],
                "amount": position["amount"],
                "entry_price": position["entry_price"],
                "current_price": current_price,
                "value": position["amount"] * current_price,
                "pnl_percentage": pnl_percentage * 100,
                "pnl_amount": pnl_amount,
                "opened_at": position["opened_at"]
            })
        
        return positions
    
    def close_position(self, symbol: str) -> Dict:
        """Close a position"""
        if symbol not in self.positions:
            return {"success": False, "error": "Position not found"}
        
        position = self.positions[symbol]
        
        # Get current price
        current_price = self._get_current_price(symbol)
        
        # Calculate realized PnL
        if position["side"] == "buy":
            pnl_percentage = (current_price - position["entry_price"]) / position["entry_price"]
        else:
            pnl_percentage = (position["entry_price"] - current_price) / position["entry_price"]
            
        pnl_amount = pnl_percentage * position["value"]
        
        # Update balance
        position_value = position["amount"] * current_price
        self.balance["free"] += position_value + pnl_amount
        
        # Record the trade
        trade = {
            "symbol": symbol,
            "side": position["side"],
            "amount": position["amount"],
            "entry_price": position["entry_price"],
            "exit_price": current_price,
            "pnl_percentage": pnl_percentage * 100,
            "pnl_amount": pnl_amount,
            "opened_at": position["opened_at"],
            "closed_at": datetime.now().isoformat(),
            "duration": (datetime.now() - datetime.fromisoformat(position["opened_at"])).total_seconds() / 3600  # in hours
        }
        
        self.executed_trades.append(trade)
        
        # Remove the position
        del self.positions[symbol]
        
        # Save state
        self.save_state()
        
        return {
            "success": True,
            "trade": trade
        }
    
    def get_ticker(self, symbol: str) -> Dict:
        """Get current ticker information"""
        current_price = self._get_current_price(symbol)
        
        return {
            "symbol": symbol,
            "price": current_price,
            "timestamp": int(time.time() * 1000)
        }
    
    def _get_current_price(self, symbol: str) -> float:
        """Get current price for a symbol"""
        # Try from external data feed first
        if self.external_data_feed:
            try:
                if hasattr(self.external_data_feed, 'get_ticker'):
                    ticker = self.external_data_feed.get_ticker(symbol)
                    price = float(ticker.get("price", 0))
                    
                    if price > 0:
                        self.price_cache[symbol] = price
                        return price
            except Exception as e:
                logger.warning(f"Error getting price from external feed: {str(e)}")
        
        # Fallback to cached price
        if symbol in self.price_cache:
            return self.price_cache[symbol]
        
        # Fallback to mock price if no price available
        # Generate realistic mock prices for popular cryptocurrencies
        base_prices = {
            "BTCUSDT": 35000,
            "ETHUSDT": 2000,
            "BNBUSDT": 300,
            "ADAUSDT": 0.5,
            "XRPUSDT": 0.5,
            "DOTUSDT": 10,
            "LINKUSDT": 15,
            "LTCUSDT": 100,
            "BCHUSDT": 300,
            "XLMUSDT": 0.3
        }
        
        # If symbol exists, return its base price with small random variation
        if symbol in base_prices:
            price = base_prices[symbol] * (1 + 0.001 * np.random.randn())
        else:
            # For unknown symbols, generate a random price between 0.1 and 1000
            price = 10 ** (np.random.random() * 4 - 1)
        
        # Cache the price
        self.price_cache[symbol] = price
        
        return price
    
    def _execute_order(self, order_id: str) -> Dict:
        """Execute a paper trading order"""
        if order_id not in self.orders:
            return {"success": False, "error": "Order not found"}
        
        order = self.orders[order_id]
        
        # Update order status
        order["status"] = "filled"
        order["updated_at"] = datetime.now().isoformat()
        
        # Create or update position
        symbol = order["symbol"]
        side = order["side"]
        amount = order["amount"]
        price = order["price"]
        value = order["value"]
        
        if side == "buy":
            # Release reserved funds
            self.balance["used"] -= value
            
            # Create/update position
            if symbol in self.positions:
                # Average down/up the position
                position = self.positions[symbol]
                
                if position["side"] == "buy":
                    # Same direction - average the entry price
                    total_value = position["value"] + value
                    total_amount = position["amount"] + amount
                    avg_price = total_value / total_amount
                    
                    position["entry_price"] = avg_price
                    position["amount"] = total_amount
                    position["value"] = total_value
                else:
                    # Opposite direction - reduce position
                    if position["amount"] > amount:
                        # Reduce the position
                        position["amount"] -= amount
                        position["value"] = position["amount"] * position["entry_price"]
                    elif position["amount"] < amount:
                        # Flip the position
                        new_amount = amount - position["amount"]
                        self.positions[symbol] = {
                            "symbol": symbol,
                            "side": "buy",
                            "amount": new_amount,
                            "entry_price": price,
                            "value": new_amount * price,
                            "opened_at": datetime.now().isoformat()
                        }
                    else:
                        # Close the position
                        del self.positions[symbol]
            else:
                # Create new position
                self.positions[symbol] = {
                    "symbol": symbol,
                    "side": "buy",
                    "amount": amount,
                    "entry_price": price,
                    "value": value,
                    "opened_at": datetime.now().isoformat()
                }
        else:  # sell
            # Create/update position
            if symbol in self.positions:
                # Update existing position
                position = self.positions[symbol]
                
                if position["side"] == "sell":
                    # Same direction - average the entry price
                    total_value = position["value"] + value
                    total_amount = position["amount"] + amount
                    avg_price = total_value / total_amount
                    
                    position["entry_price"] = avg_price
                    position["amount"] = total_amount
                    position["value"] = total_value
                else:
                    # Opposite direction - reduce position
                    if position["amount"] > amount:
                        # Reduce the position
                        position["amount"] -= amount
                        position["value"] = position["amount"] * position["entry_price"]
                    elif position["amount"] < amount:
                        # Flip the position
                        new_amount = amount - position["amount"]
                        self.positions[symbol] = {
                            "symbol": symbol,
                            "side": "sell",
                            "amount": new_amount,
                            "entry_price": price,
                            "value": new_amount * price,
                            "opened_at": datetime.now().isoformat()
                        }
                    else:
                        # Close the position
                        del self.positions[symbol]
            else:
                # Create new position
                self.positions[symbol] = {
                    "symbol": symbol,
                    "side": "sell",
                    "amount": amount,
                    "entry_price": price,
                    "value": value,
                    "opened_at": datetime.now().isoformat()
                }
            
            # Add funds from sell
            self.balance["free"] += value
        
        # Move order to history
        self.order_history.append(order)
        del self.orders[order_id]
        
        # Save state
        self.save_state()
        
        return {"success": True, "order_id": order_id}
    
    def _timeframe_to_minutes(self, timeframe: str) -> int:
        """Convert a timeframe string to minutes"""
        if timeframe.endswith('m'):
            return int(timeframe[:-1])
        elif timeframe.endswith('h'):
            return int(timeframe[:-1]) * 60
        elif timeframe.endswith('d'):
            return int(timeframe[:-1]) * 1440  # days to minutes
        elif timeframe.endswith('w'):
            return int(timeframe[:-1]) * 10080  # weeks to minutes
        else:
            return 60  # default to 1 hour
    
    def save_state(self) -> None:
        """Save the current state to disk"""
        state = {
            "balance": self.balance,
            "orders": self.orders,
            "positions": self.positions,
            "order_history": self.order_history[-100:],  # Keep only last 100 orders
            "executed_trades": self.executed_trades[-100:],  # Keep only last 100 trades
            "price_cache": self.price_cache
        }
        
        try:
            state_path = os.path.join(self.storage_dir, "paper_trading_state.json")
            with open(state_path, 'w') as f:
                json.dump(state, f, indent=2, default=str)
        except Exception as e:
            logger.error(f"Failed to save paper trading state: {str(e)}")
    
    def load_state(self) -> None:
        """Load the state from disk"""
        state_path = os.path.join(self.storage_dir, "paper_trading_state.json")
        
        if not os.path.exists(state_path):
            logger.info("No paper trading state file found, using default state")
            return
        
        try:
            with open(state_path, 'r') as f:
                state = json.load(f)
            
            self.balance = state.get("balance", self.balance)
            self.orders = state.get("orders", {})
            self.positions = state.get("positions", {})
            self.order_history = state.get("order_history", [])
            self.executed_trades = state.get("executed_trades", [])
            self.price_cache = state.get("price_cache", {})
            
            logger.info(f"Loaded paper trading state: {len(self.positions)} positions, {len(self.orders)} open orders")
        except Exception as e:
            logger.error(f"Failed to load paper trading state: {str(e)}")