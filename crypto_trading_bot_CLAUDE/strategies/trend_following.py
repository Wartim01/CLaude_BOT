"""
Trend-following strategy that uses moving averages and other indicators
to identify and follow market trends
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any

from utils.logger import setup_logger

logger = setup_logger("trend_following")

class Strategy:
    """
    Trend following strategy implementation
    
    Uses multiple technical indicators to identify trends:
    - EMA crossovers (short/medium/long term)
    - RSI for momentum and trend strength
    - MACD for trend confirmation
    - ATR for position sizing and stop placement
    """
    def __init__(self, ema_short: int = 9, ema_medium: int = 21, ema_long: int = 55,
                 rsi_period: int = 14, rsi_overbought: int = 70, rsi_oversold: int = 30,
                 atr_period: int = 14, atr_multiplier: float = 2.0):
        """
        Initialize the trend following strategy with parameters
        
        Args:
            ema_short: Short-term EMA period
            ema_medium: Medium-term EMA period
            ema_long: Long-term EMA period
            rsi_period: RSI calculation period
            rsi_overbought: RSI overbought threshold
            rsi_oversold: RSI oversold threshold
            atr_period: ATR calculation period
            atr_multiplier: Multiplier for ATR-based stops
        """
        self.ema_short = ema_short
        self.ema_medium = ema_medium
        self.ema_long = ema_long
        self.rsi_period = rsi_period
        self.rsi_overbought = rsi_overbought
        self.rsi_oversold = rsi_oversold
        self.atr_period = atr_period
        self.atr_multiplier = atr_multiplier
        
        # Performance tracking
        self.signals_generated = {}
        self.last_signals = {}
        
        logger.info(f"Trend following strategy initialized with EMA({ema_short},{ema_medium},{ema_long}), "
                    f"RSI({rsi_period}), ATR({atr_period})")
    
    def generate_signal(self, df: pd.DataFrame, symbol: str, timeframe: str) -> Dict[str, Any]:
        """
        Generate trading signal based on trend indicators
        
        Args:
            df: DataFrame with OHLCV data and indicators
            symbol: Trading symbol
            timeframe: Timeframe of the data
            
        Returns:
            Signal dictionary
        """
        # Check if we have enough data
        if len(df) < self.ema_long + 10:
            return {"direction": "NEUTRAL", "signal_strength": 0, "timeframe": timeframe}
        
        # Get the most recent data
        current_data = df.iloc[-1]
        prior_data = df.iloc[-2]
        
        # Get required indicators
        required_columns = [f'ema_{self.ema_short}', f'ema_{self.ema_medium}', f'ema_{self.ema_long}',
                           'rsi', 'macd', 'macd_signal', 'atr']
        
        if not all(col in df.columns for col in required_columns):
            logger.warning(f"Missing required indicators for trend following strategy: "
                           f"{[col for col in required_columns if col not in df.columns]}")
            
            # Try to calculate missing indicators if possible
            self._add_missing_indicators(df)
        
        # Check for EMA crossover signals
        ema_short_name = f'ema_{self.ema_short}'
        ema_med_name = f'ema_{self.ema_medium}'
        ema_long_name = f'ema_{self.ema_long}'
        
        # Current EMAs
        if ema_short_name in current_data and ema_med_name in current_data:
            ema_short_now = current_data[ema_short_name]
            ema_med_now = current_data[ema_med_name]
            ema_long_now = current_data[ema_long_name] if ema_long_name in current_data else None
            
            # Previous EMAs
            ema_short_prev = prior_data[ema_short_name]
            ema_med_prev = prior_data[ema_med_name]
            ema_long_prev = prior_data[ema_long_name] if ema_long_name in current_data else None
            
            # Check EMA crossovers (short crossing above/below medium)
            ema_crossover_up = ema_short_prev < ema_med_prev and ema_short_now > ema_med_now
            ema_crossover_down = ema_short_prev > ema_med_prev and ema_short_now < ema_med_now
            
            # Check EMA alignment (all EMAs aligned in same direction)
            ema_aligned_up = False
            ema_aligned_down = False
            
            if ema_long_now is not None:
                ema_aligned_up = ema_short_now > ema_med_now > ema_long_now
                ema_aligned_down = ema_short_now < ema_med_now < ema_long_now
        else:
            ema_crossover_up = False
            ema_crossover_down = False
            ema_aligned_up = False
            ema_aligned_down = False
        
        # Check RSI conditions
        rsi_now = current_data.get('rsi', 50)
        rsi_prev = prior_data.get('rsi', 50)
        
        rsi_oversold = rsi_now < self.rsi_oversold
        rsi_overbought = rsi_now > self.rsi_overbought
        rsi_rising = rsi_now > rsi_prev
        rsi_falling = rsi_now < rsi_prev
        
        # Check MACD conditions
        macd_now = current_data.get('macd', 0)
        macd_signal_now = current_data.get('macd_signal', 0)
        macd_prev = prior_data.get('macd', 0)
        macd_signal_prev = prior_data.get('macd_signal', 0)
        
        macd_crossover_up = macd_prev < macd_signal_prev and macd_now > macd_signal_now
        macd_crossover_down = macd_prev > macd_signal_prev and macd_now < macd_signal_now
        macd_positive = macd_now > 0
        macd_negative = macd_now < 0
        
        # Calculate ATR-based stop loss if available
        atr_stop = None
        stop_loss = None
        if 'atr' in current_data:
            atr_value = current_data['atr']
            current_price = current_data['close']
            atr_stop = atr_value * self.atr_multiplier
        
        # Determine signals
        # Initialize with NEUTRAL
        direction = "NEUTRAL"
        signal_strength = 0
        signals = []
        confidence = 0.0
        
        # Bullish signals
        bullish_signals = []
        if ema_crossover_up:
            bullish_signals.append("EMA_CROSSOVER_UP")
        if ema_aligned_up:
            bullish_signals.append("EMA_ALIGNED_UP")
        if rsi_oversold and rsi_rising:
            bullish_signals.append("RSI_OVERSOLD_RISING")
        if macd_crossover_up:
            bullish_signals.append("MACD_CROSSOVER_UP")
        if macd_positive:
            bullish_signals.append("MACD_POSITIVE")
        
        # Bearish signals
        bearish_signals = []
        if ema_crossover_down:
            bearish_signals.append("EMA_CROSSOVER_DOWN")
        if ema_aligned_down:
            bearish_signals.append("EMA_ALIGNED_DOWN")
        if rsi_overbought and rsi_falling:
            bearish_signals.append("RSI_OVERBOUGHT_FALLING")
        if macd_crossover_down:
            bearish_signals.append("MACD_CROSSOVER_DOWN")
        if macd_negative:
            bearish_signals.append("MACD_NEGATIVE")
        
        # Determine direction and strength
        if len(bullish_signals) > len(bearish_signals):
            direction = "BUY"
            signal_strength = min(5, len(bullish_signals))
            signals = bullish_signals
            confidence = min(1.0, len(bullish_signals) / 5.0)
            
            # Calculate stop loss for BUY
            if atr_stop is not None:
                stop_loss = current_price - atr_stop
                
        elif len(bearish_signals) > len(bullish_signals):
            direction = "SELL"
            signal_strength = min(5, len(bearish_signals))
            signals = bearish_signals
            confidence = min(1.0, len(bearish_signals) / 5.0)
            
            # Calculate stop loss for SELL
            if atr_stop is not None:
                stop_loss = current_price + atr_stop
        
        # Create signal
        signal = {
            "direction": direction,
            "signal_strength": signal_strength,
            "signals": signals,
            "confidence": confidence,
            "timeframe": timeframe,
            "price": current_data['close']
        }
        
        if stop_loss is not None:
            signal["stop_loss"] = stop_loss
        
        # Store the signal for performance tracking
        signal_key = f"{symbol}_{timeframe}"
        if signal_key not in self.signals_generated:
            self.signals_generated[signal_key] = []
        
        signal_record = signal.copy()
        signal_record["timestamp"] = pd.Timestamp.now().isoformat()
        
        self.signals_generated[signal_key].append(signal_record)
        self.last_signals[signal_key] = signal_record
        
        # Cap the signal history
        max_history = 100
        if len(self.signals_generated[signal_key]) > max_history:
            self.signals_generated[signal_key] = self.signals_generated[signal_key][-max_history:]
        
        return signal
    
    def check_exit(self, df: pd.DataFrame, symbol: str, direction: str, position: Dict) -> Dict:
        """
        Check if an existing position should be exited
        
        Args:
            df: DataFrame with OHLCV data and indicators
            symbol: Trading symbol
            direction: Current position direction ('BUY' or 'SELL')
            position: Position data dictionary
            
        Returns:
            Exit signal dictionary
        """
        # Check if we have enough data
        if len(df) < self.ema_long + 10:
            return {"should_exit": False}
        
        # Get the most recent data
        current_data = df.iloc[-1]
        prior_data = df.iloc[-2]
        current_price = current_data['close']
        
        # Get ema columns
        ema_short_name = f'ema_{self.ema_short}'
        ema_med_name = f'ema_{self.ema_medium}'
        
        exit_reasons = []
        
        # Check for trend reversal based on EMA crossover
        if ema_short_name in current_data and ema_med_name in current_data:
            ema_short_now = current_data[ema_short_name]
            ema_med_now = current_data[ema_med_name]
            ema_short_prev = prior_data[ema_short_name]
            ema_med_prev = prior_data[ema_med_name]
            
            if direction == "BUY":
                # For long positions, check for bearish crossover
                if ema_short_prev > ema_med_prev and ema_short_now < ema_med_now:
                    exit_reasons.append("EMA_CROSSOVER_EXIT")
            else:
                # For short positions, check for bullish crossover
                if ema_short_prev < ema_med_prev and ema_short_now > ema_med_now:
                    exit_reasons.append("EMA_CROSSOVER_EXIT")
        
        # Check for RSI divergence
        rsi_now = current_data.get('rsi', 50)
        
        if direction ==