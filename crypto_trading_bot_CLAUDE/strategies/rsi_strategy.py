"""
RSI-based trading strategy with dynamic thresholds
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Any

def calculate_rsi(data: pd.DataFrame, period: int = 14) -> pd.Series:
    """
    Calculate Relative Strength Index
    
    Args:
        data: DataFrame with OHLCV data
        period: RSI period
        
    Returns:
        Series containing RSI values
    """
    delta = data['close'].diff()
    gain = delta.where(delta > 0, 0).fillna(0)
    loss = -delta.where(delta < 0, 0).fillna(0)
    
    avg_gain = gain.rolling(window=period).mean()
    avg_loss = loss.rolling(window=period).mean()
    
    rs = avg_gain / avg_loss.replace(0, 1e-9)  # Avoid division by zero
    rsi = 100 - (100 / (1 + rs))
    
    return rsi

def adaptive_rsi_strategy(data: pd.DataFrame, positions: List = None, 
                        current_equity: float = None, **params) -> List[Dict]:
    """
    Adaptive RSI strategy with dynamic thresholds
    
    Args:
        data: DataFrame with market data
        positions: Current open positions
        current_equity: Current equity value
        **params: Additional strategy parameters
        
    Returns:
        List of trading signals
    """
    signals = []
    
    # Skip if we already have a position
    if positions:
        return signals
    
    # Get parameters or use defaults
    rsi_period = params.get('rsi_period', 14)
    base_oversold = params.get('oversold', 30)
    base_overbought = params.get('overbought', 70)
    atr_period = params.get('atr_period', 14)
    volatility_adjustment = params.get('volatility_adjustment', True)
    
    # Ensure we have enough data
    if len(data) < max(rsi_period, atr_period) + 10:
        return signals
    
    # Calculate RSI
    if 'rsi' not in data.columns:
        data['rsi'] = calculate_rsi(data, rsi_period)
    
    # Calculate ATR for volatility adjustment
    if volatility_adjustment and 'atr' not in data.columns:
        data['tr'] = np.maximum(
            data['high'] - data['low'],
            np.maximum(
                abs(data['high'] - data['close'].shift(1)),
                abs(data['low'] - data['close'].shift(1))
            )
        )
        data['atr'] = data['tr'].rolling(window=atr_period).mean()
        data['atr_percent'] = data['atr'] / data['close'] * 100
    
    # Get current values
    current = data.iloc[-1]
    previous = data.iloc[-2]
    
    # Adjust thresholds based on volatility
    oversold = base_oversold
    overbought = base_overbought
    
    if volatility_adjustment and 'atr_percent' in data.columns:
        # Calculate normal volatility range
        avg_atr = data['atr_percent'].mean()
        
        # Adjust thresholds: higher volatility = wider thresholds
        volatility_factor = current['atr_percent'] / avg_atr if avg_atr > 0 else 1.0
        
        # Limit the adjustment factor
        volatility_factor = max(0.5, min(1.5, volatility_factor))
        
        # Adjust thresholds
        oversold = base_oversold - (5 * (volatility_factor - 1))
        overbought = base_overbought + (5 * (volatility_factor - 1))
    
    # Generate buy signal when RSI crosses above oversold threshold
    if previous['rsi'] <= oversold and current['rsi'] > oversold:
        # Calculate position size - use larger size when RSI is deeply oversold
        conviction = (oversold - min(previous['rsi'], oversold)) / oversold
        size_pct = 50 + (conviction * 50)  # 50% to 100% of available capital
        
        # Calculate dynamic stop loss based on recent volatility
        stop_loss_pct = current['atr_percent'] * 1.5 if 'atr_percent' in current else 2.0
        
        # Calculate take profit - higher for stronger signals
        take_profit_pct = stop_loss_pct * (1.5 + conviction)
        
        signals.append({
            'direction': 'long',
            'size_pct': size_pct,  # Dynamic position sizing
            'stop_loss_pct': stop_loss_pct,  # Dynamic stop loss
            'take_profit_pct': take_profit_pct,  # Dynamic take profit
            'metadata': {
                'rsi': current['rsi'],
                'oversold_threshold': oversold,
                'conviction': conviction,
                'volatility': current.get('atr_percent', 0)
            }
        })
    
    return signals
