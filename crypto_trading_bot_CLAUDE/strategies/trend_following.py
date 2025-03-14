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
        
        if direction == "BUY":
            # For long positions, check for overbought conditions
            if rsi_now > self.rsi_overbought:
                exit_reasons.append("RSI_OVERBOUGHT_EXIT")
        else:
            # For short positions, check for oversold conditions
            if rsi_now < self.rsi_oversold:
                exit_reasons.append("RSI_OVERSOLD_EXIT")
                
        # Check for MACD reversal
        macd_now = current_data.get('macd', 0)
        macd_signal_now = current_data.get('macd_signal', 0)
        macd_prev = prior_data.get('macd', 0)
        macd_signal_prev = prior_data.get('macd_signal', 0)
        
        if direction == "BUY":
            # For long positions, check for bearish MACD crossover
            if macd_prev > macd_signal_prev and macd_now < macd_signal_now:
                exit_reasons.append("MACD_CROSSOVER_EXIT")
        else:
            # For short positions, check for bullish MACD crossover
            if macd_prev < macd_signal_prev and macd_now > macd_signal_now:
                exit_reasons.append("MACD_CROSSOVER_EXIT")
        
        # Check trailing stop if applicable
        if "entry_price" in position and "stop_loss" in position:
            entry_price = position["entry_price"]
            stop_loss = position["stop_loss"]
            
            # For long positions
            if direction == "BUY":
                # Check if price is below stop loss
                if current_price <= stop_loss:
                    exit_reasons.append("STOP_LOSS_TRIGGERED")
            # For short positions
            else:
                # Check if price is above stop loss
                if current_price >= stop_loss:
                    exit_reasons.append("STOP_LOSS_TRIGGERED")
        
        # Check if we have any reason to exit
        should_exit = len(exit_reasons) > 0
        
        # Create exit signal
        exit_signal = {
            "should_exit": should_exit,
            "reasons": exit_reasons,
            "timeframe": df.attrs.get("timeframe", "unknown"),
            "price": current_price
        }
        
        return exit_signal
    
    def _add_missing_indicators(self, df: pd.DataFrame) -> None:
        """
        Calculate any missing indicators required for the strategy
        
        Args:
            df: DataFrame with OHLCV data
        """
        # Calculate EMAs if missing
        ema_short_name = f'ema_{self.ema_short}'
        if ema_short_name not in df.columns:
            df[ema_short_name] = df['close'].ewm(span=self.ema_short, adjust=False).mean()
            
        ema_med_name = f'ema_{self.ema_medium}'
        if ema_med_name not in df.columns:
            df[ema_med_name] = df['close'].ewm(span=self.ema_medium, adjust=False).mean()
            
        ema_long_name = f'ema_{self.ema_long}'
        if ema_long_name not in df.columns:
            df[ema_long_name] = df['close'].ewm(span=self.ema_long, adjust=False).mean()
        
        # Calculate RSI if missing
        if 'rsi' not in df.columns:
            delta = df['close'].diff()
            gain = delta.where(delta > 0, 0).rolling(window=self.rsi_period).mean()
            loss = -delta.where(delta < 0, 0).rolling(window=self.rsi_period).mean()
            rs = gain / loss
            df['rsi'] = 100 - (100 / (1 + rs))
        
        # Calculate MACD if missing
        if 'macd' not in df.columns or 'macd_signal' not in df.columns:
            ema12 = df['close'].ewm(span=12, adjust=False).mean()
            ema26 = df['close'].ewm(span=26, adjust=False).mean()
            df['macd'] = ema12 - ema26
            df['macd_signal'] = df['macd'].ewm(span=9, adjust=False).mean()
        
        # Calculate ATR if missing
        if 'atr' not in df.columns:
            high_low = df['high'] - df['low']
            high_close = (df['high'] - df['close'].shift()).abs()
            low_close = (df['low'] - df['close'].shift()).abs()
            tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
            df['atr'] = tr.rolling(window=self.atr_period).mean()
    
    def update_stop_loss(self, df: pd.DataFrame, position: Dict) -> float:
        """
        Update stop loss level for trailing stops
        
        Args:
            df: DataFrame with OHLCV data and indicators
            position: Current position information
            
        Returns:
            Updated stop loss level
        """
        # Get current price and direction
        current_price = df.iloc[-1]['close']
        direction = position.get("direction", "NEUTRAL")
        
        # Get current stop loss
        current_stop = position.get("stop_loss")
        if current_stop is None:
            return None
            
        # Get ATR value if available
        atr_value = df.iloc[-1].get('atr')
        if atr_value is None:
            return current_stop  # No update if ATR is not available
        
        # Calculate new stop loss
        new_stop = current_stop
        
        if direction == "BUY":
            # For long positions, move stop up if price rises
            trailing_stop = current_price - (atr_value * self.atr_multiplier)
            if trailing_stop > current_stop:
                new_stop = trailing_stop
        else:
            # For short positions, move stop down if price falls
            trailing_stop = current_price + (atr_value * self.atr_multiplier)
            if trailing_stop < current_stop:
                new_stop = trailing_stop
        
        return new_stop