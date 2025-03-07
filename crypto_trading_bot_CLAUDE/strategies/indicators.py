"""
Technical indicators used by trading strategies
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Union, Tuple

def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add common technical indicators to a DataFrame
    
    Args:
        df: DataFrame with OHLCV data
        
    Returns:
        DataFrame with indicators
    """
    # Make a copy to avoid modifying the original DataFrame
    df = df.copy()
    
    # Moving averages
    df['sma_20'] = df['close'].rolling(window=20).mean()
    df['sma_50'] = df['close'].rolling(window=50).mean()
    df['sma_200'] = df['close'].rolling(window=200).mean()
    
    df['ema_9'] = df['close'].ewm(span=9, adjust=False).mean()
    df['ema_21'] = df['close'].ewm(span=21, adjust=False).mean()
    
    # MACD
    df['ema_12'] = df['close'].ewm(span=12, adjust=False).mean()
    df['ema_26'] = df['close'].ewm(span=26, adjust=False).mean()
    df['macd'] = df['ema_12'] - df['ema_26']
    df['macd_signal'] = df['macd'].ewm(span=9, adjust=False).mean()
    df['macd_hist'] = df['macd'] - df['macd_signal']
    
    # RSI
    delta = df['close'].diff()
    gain = delta.where(delta > 0, 0).rolling(window=14).mean()
    loss = -delta.where(delta < 0, 0).rolling(window=14).mean()
    
    rs = gain / loss
    df['rsi'] = 100 - (100 / (1 + rs))
    
    # Bollinger Bands
    df['bb_middle'] = df['close'].rolling(window=20).mean()
    df['bb_std'] = df['close'].rolling(window=20).std()
    df['bb_upper'] = df['bb_middle'] + 2 * df['bb_std']
    df['bb_lower'] = df['bb_middle'] - 2 * df['bb_std']
    df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / df['bb_middle']
    
    # ATR (Average True Range)
    high_low = df['high'] - df['low']
    high_close = (df['high'] - df['close'].shift()).abs()
    low_close = (df['low'] - df['close'].shift()).abs()
    
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    df['atr'] = tr.rolling(window=14).mean()
    
    # Volume-based indicators
    df['volume_sma'] = df['volume'].rolling(window=20).mean()
    df['volume_ratio'] = df['volume'] / df['volume_sma']
    
    # OBV (On-Balance Volume)
    df['obv'] = (np.sign(df['close'].diff()) * df['volume']).fillna(0).cumsum()
    
    return df

def calculate_ema(df: pd.DataFrame, periods: List[int] = [9, 21, 50, 200]) -> Dict[int, pd.Series]:
    """
    Calculate EMA for multiple periods
    
    Args:
        df: DataFrame with 'close' column
        periods: List of periods to calculate
        
    Returns:
        Dictionary of EMAs
    """
    result = {}
    for period in periods:
        result[period] = df['close'].ewm(span=period, adjust=False).mean()
        result[period].name = f"ema_{period}"
    return result

def calculate_macd(df: pd.DataFrame, fast: int = 12, slow: int = 26, signal: int = 9) -> Dict[str, pd.Series]:
    """
    Calculate MACD indicator
    
    Args:
        df: DataFrame with 'close' column
        fast: Fast EMA period
        slow: Slow EMA period
        signal: Signal line period
        
    Returns:
        Dictionary with MACD components
    """
    ema_fast = df['close'].ewm(span=fast, adjust=False).mean()
    ema_slow = df['close'].ewm(span=slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    histogram = macd_line - signal_line
    
    return {
        "macd": macd_line,
        "signal": signal_line,
        "histogram": histogram
    }

def calculate_rsi(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """
    Calculate RSI indicator
    
    Args:
        df: DataFrame with 'close' column
        period: RSI period
        
    Returns:
        RSI series
    """
    delta = df['close'].diff()
    gain = delta.where(delta > 0, 0).rolling(window=period).mean()
    loss = -delta.where(delta < 0, 0).rolling(window=period).mean()
    
    rs = gain / loss
    return 100 - (100 / (1 + rs))

def calculate_bollinger_bands(df: pd.DataFrame, period: int = 20, std_dev: float = 2.0) -> Dict[str, pd.Series]:
    """
    Calculate Bollinger Bands
    
    Args:
        df: DataFrame with 'close' column
        period: Moving average period
        std_dev: Standard deviation factor
        
    Returns:
        Dictionary with Bollinger Band components
    """
    middle = df['close'].rolling(window=period).mean()
    std = df['close'].rolling(window=period).std()
    
    upper = middle + (std * std_dev)
    lower = middle - (std * std_dev)
    
    # Calculate bandwidth and %B
    bandwidth = (upper - lower) / middle
    percent_b = (df['close'] - lower) / (upper - lower)
    
    return {
        "upper": upper,
        "middle": middle,
        "lower": lower,
        "bandwidth": bandwidth,
        "percent_b": percent_b
    }

def calculate_atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """
    Calculate Average True Range (ATR)
    
    Args:
        df: DataFrame with OHLC data
        period: ATR period
        
    Returns:
        ATR series
    """
    high_low = df['high'] - df['low']
    high_close = (df['high'] - df['close'].shift()).abs()
    low_close = (df['low'] - df['close'].shift()).abs()
    
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    return tr.rolling(window=period).mean()

def calculate_stochastic(df: pd.DataFrame, k_period: int = 14, d_period: int = 3) -> Dict[str, pd.Series]:
    """
    Calculate Stochastic Oscillator
    
    Args:
        df: DataFrame with OHLC data
        k_period: %K period
        d_period: %D period
        
    Returns:
        Dictionary with Stochastic components
    """
    low_min = df['low'].rolling(window=k_period).min()
    high_max = df['high'].rolling(window=k_period).max()
    
    # Calculate %K
    k = 100 * ((df['close'] - low_min) / (high_max - low_min))
    
    # Calculate %D (SMA of %K)
    d = k.rolling(window=d_period).mean()
    
    return {
        "k": k,
        "d": d
    }

def calculate_obv(df: pd.DataFrame) -> pd.Series:
    """
    Calculate On-Balance Volume (OBV)
    
    Args:
        df: DataFrame with 'close' and 'volume' columns
        
    Returns:
        OBV series
    """
    return (np.sign(df['close'].diff()) * df['volume']).fillna(0).cumsum()

def calculate_vwap(df: pd.DataFrame) -> pd.Series:
    """
    Calculate Volume-Weighted Average Price (VWAP)
    
    Args:
        df: DataFrame with OHLCV data
        
    Returns:
        VWAP series
    """
    # Calculate typical price
    typical_price = (df['high'] + df['low'] + df['close']) / 3
    
    # Calculate VWAP
    vwap = (typical_price * df['volume']).cumsum() / df['volume'].cumsum()
    
    return vwap
