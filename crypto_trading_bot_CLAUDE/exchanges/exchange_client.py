"""
Abstract base class for exchange client implementations
"""
import abc
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime

class ExchangeClient(abc.ABC):
    """
    Abstract base class defining the interface for exchange API clients
    """
    
    @abc.abstractmethod
    def test_connection(self) -> bool:
        """
        Test the connection to the exchange
        
        Returns:
            Boolean indicating success
        """
        pass
    
    @abc.abstractmethod
    def get_balance(self) -> Dict:
        """
        Get account balance
        
        Returns:
            Dictionary with account balance information
        """
        pass
    
    @abc.abstractmethod
    def fetch_ohlcv(self, symbol: str, timeframe: str, limit: int = 100) -> List:
        """
        Fetch OHLCV (Open, High, Low, Close, Volume) candlestick data
        
        Args:
            symbol: Symbol to fetch data for
            timeframe: Timeframe interval
            limit: Number of candles to fetch
            
        Returns:
            List of OHLCV candles
        """
        pass
    
    @abc.abstractmethod
    def create_order(self, symbol: str, order_type: str, side: str, amount: float, price: float = None) -> Dict:
        """
        Create a new order
        
        Args:
            symbol: Symbol to trade
            order_type: Order type (market, limit, etc.)
            side: Order side (buy or sell)
            amount: Order amount
            price: Order price (for limit orders)
            
        Returns:
            Order information
        """
        pass
    
    @abc.abstractmethod
    def get_order(self, order_id: str, symbol: str = None) -> Dict:
        """
        Get order information
        
        Args:
            order_id: Order ID
            symbol: Symbol (optional for some exchanges)
            
        Returns:
            Order information
        """
        pass
    
    @abc.abstractmethod
    def cancel_order(self, order_id: str, symbol: str = None) -> Dict:
        """
        Cancel an order
        
        Args:
            order_id: Order ID
            symbol: Symbol (optional for some exchanges)
            
        Returns:
            Cancellation result
        """
        pass
    
    @abc.abstractmethod
    def get_open_positions(self) -> List[Dict]:
        """
        Get open positions
        
        Returns:
            List of open positions
        """
        pass
    
    @abc.abstractmethod
    def close_position(self, symbol: str) -> Dict:
        """
        Close a position
        
        Args:
            symbol: Symbol to close position for
            
        Returns:
            Close position result
        """
        pass
    
    @abc.abstractmethod
    def get_ticker(self, symbol: str) -> Dict:
        """
        Get current ticker information
        
        Args:
            symbol: Symbol to get ticker for
            
        Returns:
            Ticker information
        """
        pass

    @abc.abstractmethod
    def check_connection(self) -> bool:
        """
        Check connection to the exchange using a ping or test API call
        
        Returns:
            Boolean indicating success
        """
        pass
