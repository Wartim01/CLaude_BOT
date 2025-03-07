"""
Data management system for retrieving and processing market data
"""
import os
import json
import time
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Union, Any
from datetime import datetime, timedelta

from config.config import DATA_DIR
from utils.logger import setup_logger
from exchanges.exchange_client import ExchangeClient
from strategies.indicators import add_indicators

logger = setup_logger("data_manager")

class DataManager:
    """
    Manages market data collection, caching, and processing
    """
    def __init__(self, exchange_client: ExchangeClient):
        """
        Initialize the data manager
        
        Args:
            exchange_client: Exchange client instance
        """
        self.exchange = exchange_client
        
        # Data caching system
        self.cache = {}  # {(symbol, timeframe): {"data": df, "last_update": timestamp}}
        self.cache_expiry = 60  # Cache expiry in seconds
        
        # Data directory
        self.data_dir = os.path.join(DATA_DIR, "market_data")
        os.makedirs(self.data_dir, exist_ok=True)
        
        logger.info("Data manager initialized")
    
    def get_market_data(self, symbol: str, timeframe: str, 
                     limit: int = 500, include_indicators: bool = True,
                     use_cache: bool = True) -> Optional[pd.DataFrame]:
        """
        Get market data for a symbol and timeframe
        
        Args:
            symbol: Trading pair (e.g., 'BTCUSDT')
            timeframe: Timeframe (e.g., '1h', '4h', '1d')
            limit: Number of candles to fetch
            include_indicators: Whether to include technical indicators
            use_cache: Whether to use cached data
            
        Returns:
            DataFrame with OHLCV data and indicators (if requested)
        """
        cache_key = (symbol, timeframe)
        current_time = time.time()
        
        # Check if we have valid cached data
        if use_cache and cache_key in self.cache:
            cache_entry = self.cache[cache_key]
            cache_age = current_time - cache_entry["last_update"]
            
            if cache_age < self.cache_expiry:
                logger.debug(f"Using cached data for {symbol} {timeframe}")
                return cache_entry["data"]
        
        try:
            logger.info(f"Fetching market data for {symbol} {timeframe}")
            
            # Fetch data from exchange
            ohlcv = self.exchange.fetch_ohlcv(symbol, timeframe, limit)
            
            if not ohlcv or len(ohlcv) == 0:
                logger.warning(f"No data received for {symbol} {timeframe}")
                return None
            
            # Convert to DataFrame
            df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            
            # Convert timestamp to datetime
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df.set_index('timestamp', inplace=True)
            
            # Add indicators if requested
            if include_indicators:
                df = self._add_indicators(df)
            
            # Update cache
            self.cache[cache_key] = {
                "data": df,
                "last_update": current_time
            }
            
            return df
            
        except Exception as e:
            logger.error(f"Error getting market data for {symbol} {timeframe}: {str(e)}")
            return None
    
    def get_multi_timeframe_data(self, symbol: str, timeframes: List[str], 
                               limit: int = 500, include_indicators: bool = True,
                               use_cache: bool = True) -> Dict[str, pd.DataFrame]:
        """
        Get market data for a symbol across multiple timeframes
        
        Args:
            symbol: Trading pair (e.g., 'BTCUSDT')
            timeframes: List of timeframes (e.g., ['1h', '4h', '1d'])
            limit: Number of candles to fetch
            include_indicators: Whether to include technical indicators
            use_cache: Whether to use cached data
            
        Returns:
            Dictionary mapping timeframes to DataFrames
        """
        result = {}
        
        for timeframe in timeframes:
            df = self.get_market_data(
                symbol=symbol, 
                timeframe=timeframe, 
                limit=limit,
                include_indicators=include_indicators,
                use_cache=use_cache
            )
            
            if df is not None:
                result[timeframe] = df
        
        return result
    
    def _add_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add technical indicators to the DataFrame
        
        Args:
            df: DataFrame with OHLCV data
            
        Returns:
            DataFrame with added indicators
        """
        try:
            # Use the indicators module to add standard indicators
            df = add_indicators(df)
            return df
        except Exception as e:
            logger.error(f"Error adding indicators: {str(e)}")
            return df
    
    def save_market_data(self, symbol: str, timeframe: str, data: pd.DataFrame) -> bool:
        """
        Save market data to disk
        
        Args:
            symbol: Trading pair symbol
            timeframe: Timeframe of the data
            data: DataFrame to save
            
        Returns:
            Success status
        """
        try:
            # Create directory for this symbol if it doesn't exist
            symbol_dir = os.path.join(self.data_dir, symbol)
            os.makedirs(symbol_dir, exist_ok=True)
            
            # Save to CSV
            file_path = os.path.join(symbol_dir, f"{symbol}_{timeframe}.csv")
            data.to_csv(file_path)
            
            logger.info(f"Saved market data to {file_path}")
            return True
        except Exception as e:
            logger.error(f"Error saving market data: {str(e)}")
            return False
    
    def load_market_data(self, symbol: str, timeframe: str) -> Optional[pd.DataFrame]:
        """
        Load market data from disk
        
        Args:
            symbol: Trading pair symbol
            timeframe: Timeframe of the data
            
        Returns:
            DataFrame with market data, or None if not found
        """
        try:
            file_path = os.path.join(self.data_dir, symbol, f"{symbol}_{timeframe}.csv")
            
            if not os.path.exists(file_path):
                logger.warning(f"No saved data found for {symbol} {timeframe}")
                return None
            
            # Load from CSV
            df = pd.read_csv(file_path, index_col='timestamp', parse_dates=True)
            
            logger.info(f"Loaded market data from {file_path}")
            return df
        except Exception as e:
            logger.error(f"Error loading market data: {str(e)}")
            return None
    
    def clear_cache(self, symbol: Optional[str] = None, timeframe: Optional[str] = None):
        """
        Clear the data cache
        
        Args:
            symbol: Symbol to clear (all symbols if None)
            timeframe: Timeframe to clear (all timeframes if None)
        """
        if symbol is None and timeframe is None:
            # Clear all cache
            self.cache = {}
            logger.info("Cleared all cached data")
        else:
            # Clear specific cache entries
            keys_to_remove = []
            for key in self.cache.keys():
                cache_symbol, cache_timeframe = key
                
                if (symbol is None or symbol == cache_symbol) and (timeframe is None or timeframe == cache_timeframe):
                    keys_to_remove.append(key)
            
            for key in keys_to_remove:
                del self.cache[key]
            
            logger.info(f"Cleared cache for {symbol or 'all symbols'} {timeframe or 'all timeframes'}")
