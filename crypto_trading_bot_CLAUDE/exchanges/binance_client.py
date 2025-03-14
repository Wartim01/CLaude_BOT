"""
Binance exchange client implementation
"""
import os
import json
import time
import hmac
import hashlib
import requests
from urllib.parse import urlencode
from typing import Dict, List, Optional, Any
from datetime import datetime

from utils.logger import setup_logger
from exchanges.exchange_client import ExchangeClient
from config.config import DATA_DIR

logger = setup_logger("binance_client")

class BinanceClient(ExchangeClient):
    """
    Implementation of the exchange client interface for Binance
    """
    def __init__(self, api_key: str, secret_key: str, testnet: bool = False):
        """
        Initialize Binance client
        
        Args:
            api_key: Binance API key
            api_secret: Binance API secret
            testnet: Use testnet (sandbox) environment
            **kwargs: Additional parameters
        """
        self.api_key = api_key
        self.api_secret = secret_key
        self.testnet = testnet
        
        # API URLs
        if testnet:
            self.base_url = "https://testnet.binance.vision/api"
        else:
            self.base_url = "https://api.binance.com/api"
        
        # API version
        self.api_version = "v3"
        
        # Cache for price data
        self.price_cache = {}
        self.cache_expiry = 10  # seconds
        self.cache_timestamps = {}
        
        # Request rate limiting
        self.last_request_timestamp = 0
        self.request_interval = 0.1  # minimum time between requests in seconds
        
        # Cache directory
        self.cache_dir = os.path.join(DATA_DIR, "binance_cache")
        os.makedirs(self.cache_dir, exist_ok=True)
        
        logger.info(f"Binance client initialized (testnet: {testnet})")
        
        # Test connection
        self.test_connection()
    
    def test_connection(self) -> bool:
        """Test connection to Binance API"""
        try:
            self.client.ping()  # Assuming the client provides a ping method
            return True
        except Exception as e:
            logger.error(f"Failed to connect to Binance API: {str(e)}")
            return False
    
    def get_balance(self) -> Dict:
        """Get account balance"""
        try:
            response = self._private_request("GET", f"{self.api_version}/account")
            
            balances = {}
            total_balance_usdt = 0.0
            
            # Process balances
            for balance in response.get("balances", []):
                asset = balance.get("asset", "")
                free = float(balance.get("free", 0))
                locked = float(balance.get("locked", 0))
                
                # Only include non-zero balances
                if free > 0 or locked > 0:
                    balances[asset] = {
                        "free": free,
                        "locked": locked,
                        "total": free + locked
                    }
                    
                    # Convert to USDT for estimating total balance
                    if asset != "USDT":
                        try:
                            # Get price from cache or fetch it
                            price = self.get_ticker(f"{asset}USDT").get("price", 0)
                            asset_value = (free + locked) * price
                            total_balance_usdt += asset_value
                        except:
                            # Skip assets that can't be converted to USDT
                            pass
                    else:
                        # Add USDT directly to total
                        total_balance_usdt += free + locked
            
            return {
                "balances": balances,
                "total_usdt": total_balance_usdt,
                "free_usdt": balances.get("USDT", {}).get("free", 0),
                "timestamp": int(time.time() * 1000)
            }
            
        except Exception as e:
            logger.error(f"Failed to get account balance: {str(e)}")
            return {
                "error": str(e),
                "balances": {},
                "total_usdt": 0,
                "free_usdt": 0,
                "timestamp": int(time.time() * 1000)
            }
    
    def fetch_ohlcv(self, symbol: str, timeframe: str, limit: int = 100) -> List:
        """Fetch OHLCV candlestick data"""
        try:
            # Convert timeframe to Binance interval format
            interval = self._timeframe_to_interval(timeframe)
            
            # Check cache first
            cache_key = f"{symbol}_{interval}_{limit}"
            if cache_key in self.cache_timestamps:
                cache_time = self.cache_timestamps[cache_key]
                if time.time() - cache_time < self.cache_expiry and cache_key in self.price_cache:
                    return self.price_cache[cache_key]
            
            # Prepare request parameters
            params = {
                "symbol": symbol,
                "interval": interval,
                "limit": limit
            }
            
            response = self._public_request("GET", f"{self.api_version}/klines", params)
            
            # Process response
            ohlcv_data = []
            for candle in response:
                # Binance format: [timestamp, open, high, low, close, volume, ...]
                ohlcv_data.append([
                    int(candle[0]),  # timestamp
                    float(candle[1]),  # open
                    float(candle[2]),  # high
                    float(candle[3]),  # low
                    float(candle[4]),  # close
                    float(candle[5])   # volume
                ])
            
            # Update cache
            self.price_cache[cache_key] = ohlcv_data
            self.cache_timestamps[cache_key] = time.time()
            
            return ohlcv_data
            
        except Exception as e:
            logger.error(f"Failed to fetch OHLCV data for {symbol} {timeframe}: {str(e)}")
            return []
    
    def create_order(self, symbol: str, order_type: str, side: str, amount: float, price: float = None) -> Dict:
        """Create a new order"""
        try:
            if order_type.lower() == "market":
                order = self.client.create_order(
                    symbol=symbol,
                    side=side,
                    type="MARKET",
                    quantity=amount
                )
            elif order_type.lower() == "limit":
                order = self.client.create_order(
                    symbol=symbol,
                    side=side,
                    type="LIMIT",
                    quantity=amount,
                    price=price,
                    timeInForce="GTC"
                )
            else:
                raise ValueError(f"Unsupported order type: {order_type}")
            return order
        except Exception as e:
            logger.error(f"Failed to create {order_type} {side} order for {symbol}: {str(e)}")
            return {
                "success": False,
                "error": str(e)
            }
    
    def get_order(self, order_id: str, symbol: str = None) -> Dict:
        """Get order information"""
        if symbol is None:
            logger.error("Symbol is required for Binance get_order")
            return {"success": False, "error": "Symbol is required"}
        
        try:
            params = {
                "symbol": symbol,
                "orderId": order_id,
                "timestamp": int(time.time() * 1000)
            }
            
            response = self._private_request("GET", f"{self.api_version}/order", params)
            
            return {
                "success": True,
                "order": {
                    "id": response.get("orderId"),
                    "symbol": response.get("symbol"),
                    "price": float(response.get("price", 0)),
                    "amount": float(response.get("origQty", 0)),
                    "executed": float(response.get("executedQty", 0)),
                    "side": response.get("side", "").lower(),
                    "status": response.get("status", ""),
                    "type": response.get("type", "").lower(),
                    "timestamp": response.get("time", 0)
                }
            }
            
        except Exception as e:
            logger.error(f"Failed to get order {order_id} for {symbol}: {str(e)}")
            return {
                "success": False,
                "error": str(e)
            }
    
    def cancel_order(self, order_id: str, symbol: str = None) -> Dict:
        """Cancel an order"""
        if symbol is None:
            logger.error("Symbol is required for Binance cancel_order")
            return {"success": False, "error": "Symbol is required"}
        
        try:
            params = {
                "symbol": symbol,
                "orderId": order_id,
                "timestamp": int(time.time() * 1000)
            }
            
            response = self._private_request("DELETE", f"{self.api_version}/order", params)
            
            return {
                "success": True,
                "order_id": response.get("orderId"),
                "symbol": response.get("symbol"),
                "status": "canceled"
            }
            
        except Exception as e:
            logger.error(f"Failed to cancel order {order_id} for {symbol}: {str(e)}")
            return {
                "success": False,
                "error": str(e)
            }
    
    def get_open_positions(self) -> List[Dict]:
        """Get open positions"""
        try:
            # For spot trading, open positions are represented by non-zero balances
            account_info = self._private_request("GET", f"{self.api_version}/account")
            
            positions = []
            
            for balance in account_info.get("balances", []):
                asset = balance.get("asset")
                free = float(balance.get("free", 0))
                locked = float(balance.get("locked", 0))
                
                # Skip zero balances and stablecoins
                if (free > 0 or locked > 0) and asset not in ["USDT", "USDC", "BUSD", "DAI"]:
                    # Get current price in USDT if possible
                    current_price = 0
                    try:
                        ticker = self.get_ticker(f"{asset}USDT")
                        current_price = float(ticker.get("price", 0))
                    except:
                        pass
                    
                    positions.append({
                        "symbol": f"{asset}USDT",
                        "asset": asset,
                        "amount": free + locked,
                        "free": free,
                        "locked": locked,
                        "current_price": current_price,
                        "value_usdt": (free + locked) * current_price if current_price > 0 else 0
                    })
            
            return positions
            
        except Exception as e:
            logger.error(f"Failed to get open positions: {str(e)}")
            return []
    
    def close_position(self, symbol: str) -> Dict:
        """Close a position (market sell)"""
        try:
            # For spot, closing a position means creating a market sell order
            # First, get the available balance
            base_asset = symbol.replace("USDT", "")
            account_info = self._private_request("GET", f"{self.api_version}/account")
            
            # Find the balance for the base asset
            asset_balance = 0
            for balance in account_info.get("balances", []):
                if balance.get("asset") == base_asset:
                    asset_balance = float(balance.get("free", 0))
                    break
            
            if asset_balance <= 0:
                return {
                    "success": False,
                    "error": f"No {base_asset} balance available"
                }
            
            # Create market sell order
            params = {
                "symbol": symbol,
                "side": "SELL",
                "type": "MARKET",
                "quantity": asset_balance,
                "timestamp": int(time.time() * 1000)
            }
            
            response = self._private_request("POST", f"{self.api_version}/order", params)
            
            return {
                "success": True,
                "order_id": response.get("orderId"),
                "symbol": response.get("symbol"),
                "amount": float(response.get("origQty", 0)),
                "side": "sell",
                "type": "market",
                "timestamp": response.get("transactTime", 0)
            }
            
        except Exception as e:
            logger.error(f"Failed to close position for {symbol}: {str(e)}")
            return {
                "success": False,
                "error": str(e)
            }
    
    def get_ticker(self, symbol: str) -> Dict:
        """Get current ticker information"""
        try:
            # Check cache
            cache_key = f"ticker_{symbol}"
            if cache_key in self.cache_timestamps:
                cache_time = self.cache_timestamps[cache_key]
                if time.time() - cache_time < self.cache_expiry and cache_key in self.price_cache:
                    return self.price_cache[cache_key]
            
            params = {"symbol": symbol}
            response = self._public_request("GET", f"{self.api_version}/ticker/price", params)
            
            result = {
                "symbol": response.get("symbol", symbol),
                "price": float(response.get("price", 0)),
                "timestamp": int(time.time() * 1000)
            }
            
            # Update cache
            self.price_cache[cache_key] = result
            self.cache_timestamps[cache_key] = time.time()
            
            return result
            
        except Exception as e:
            logger.error(f"Failed to get ticker for {symbol}: {str(e)}")
            return {
                "symbol": symbol,
                "price": 0,
                "error": str(e),
                "timestamp": int(time.time() * 1000)
            }
    
    def _public_request(self, method: str, endpoint: str, params: Dict = None) -> Any:
        """Make a public API request"""
        url = f"{self.base_url}/{endpoint}"
        
        # Apply rate limiting
        self._apply_rate_limit()
        
        # Make request
        try:
            if method == "GET":
                if params:
                    url += f"?{urlencode(params)}"
                response = requests.get(url)
            elif method == "POST":
                response = requests.post(url, json=params)
            elif method == "DELETE":
                response = requests.delete(url, json=params)
            else:
                raise ValueError(f"Unsupported method: {method}")
            
            # Check response
            response.raise_for_status()
            
            return response.json()
        except requests.exceptions.RequestException as e:
            logger.error(f"Request error: {str(e)}")
            raise
    
    def _private_request(self, method: str, endpoint: str, params: Dict = None) -> Any:
        """Make a private API request with authentication"""
        if not self.api_key or not self.api_secret:
            raise ValueError("API key and secret are required for private requests")
        
        # Ensure we have params
        if params is None:
            params = {}
        
        # Add timestamp if not provided
        if "timestamp" not in params:
            params["timestamp"] = int(time.time() * 1000)
        
        # Create signature
        query_string = urlencode(params)
        signature = hmac.new(
            self.api_secret.encode("utf-8"),
            query_string.encode("utf-8"),
            hashlib.sha256
        ).hexdigest()
        
        # Add signature to params
        params["signature"] = signature
        
        # Create URL
        url = f"{self.base_url}/{endpoint}"
        
        # Apply rate limiting
        self._apply_rate_limit()
        
        # Make request
        try:
            headers = {"X-MBX-APIKEY": self.api_key}
            
            if method == "GET":
                response = requests.get(url, params=params, headers=headers)
            elif method == "POST":
                response = requests.post(url, data=params, headers=headers)
            elif method == "DELETE":
                response = requests.delete(url, params=params, headers=headers)
            else:
                raise ValueError(f"Unsupported method: {method}")
            
            # Check response
            response.raise_for_status()
            
            return response.json()
        except requests.exceptions.RequestException as e:
            logger.error(f"Request error: {str(e)}")
            raise
    
    def _apply_rate_limit(self) -> None:
        """Apply rate limiting to avoid API limits"""
        current_time = time.time()
        time_since_last = current_time - self.last_request_timestamp
        
        if time_since_last < self.request_interval:
            sleep_time = self.request_interval - time_since_last
            time.sleep(sleep_time)
        
        self.last_request_timestamp = time.time()
    
    def _timeframe_to_interval(self, timeframe: str) -> str:
        """Convert a timeframe string to Binance interval format"""
        # Binance intervals: 1m, 3m, 5m, 15m, 30m, 1h, 2h, 4h, 6h, 8h, 12h, 1d, 3d, 1w, 1M
        return timeframe.lower()
    
    def check_connection(self) -> bool:
        return self.test_connection()
