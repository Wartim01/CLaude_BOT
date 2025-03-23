"""
Logger setup and utilities
"""
import os
import logging
import logging.handlers
import json
from datetime import datetime
from typing import Dict, Optional

from config.config import LOG_DIR

# Ensure log directory exists
os.makedirs(LOG_DIR, exist_ok=True)

# Custom formatter with color support for console
class ColorFormatter(logging.Formatter):
    # ANSI color codes
    COLORS = {
        'DEBUG': '\033[94m',  # Blue
        'INFO': '\033[92m',   # Green
        'WARNING': '\033[93m', # Yellow
        'ERROR': '\033[91m',   # Red
        'CRITICAL': '\033[91m\033[1m', # Bright Red
        'THRESHOLD': '\033[95m', # Purple for threshold adjustments
        'RESET': '\033[0m'     # Reset
    }

    def format(self, record):
        if not hasattr(record, 'levelname_colored') and record.levelname in self.COLORS:
            record.levelname_colored = f"{self.COLORS[record.levelname]}{record.levelname}{self.COLORS['RESET']}"
        else:
            record.levelname_colored = record.levelname
        
        return super().format(record)

# Cache for loggers
loggers = {}

def setup_logger(name, level=logging.INFO):
    """Configure and return a logger with the given name"""
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    # Prevent duplicate handlers
    if logger.handlers:
        return logger
    
    # Create log directory if it doesn't exist
    os.makedirs(LOG_DIR, exist_ok=True)
    
    # File handler for regular logs
    file_handler = logging.FileHandler(
        os.path.join(LOG_DIR, f"{name.lower().replace(' ', '_')}.log")
    )
    file_handler.setLevel(level)
    
    # Create special handler for threshold adjustments
    if name.lower() == "trading_bot":
        threshold_handler = logging.FileHandler(
            os.path.join(LOG_DIR, "threshold_adjustments.log")
        )
        threshold_handler.setLevel(logging.INFO)
        threshold_handler.setFormatter(logging.Formatter('%(asctime)s - %(message)s'))
        
        # Only log threshold adjustments to this file
        class ThresholdFilter(logging.Filter):
            def filter(self, record):
                return "THRESHOLD ADJUSTED:" in record.getMessage()
        
        threshold_handler.addFilter(ThresholdFilter())
        logger.addHandler(threshold_handler)
    
    # Create formatter for file handler
    file_formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Create formatter for console with colors
    console_formatter = ColorFormatter(
        '%(asctime)s - %(name)s - %(levelname_colored)s - %(message)s'
    )
    
    # Create file handler
    log_file = os.path.join(LOG_DIR, f"{name}.log")
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(level)
    file_handler.setFormatter(file_formatter)
    
    # Create rotating file handler for full logs
    full_log_file = os.path.join(LOG_DIR, "full_log.log")
    rotating_handler = logging.handlers.RotatingFileHandler(
        full_log_file, maxBytes=10*1024*1024, backupCount=5
    )
    rotating_handler.setLevel(level)
    rotating_handler.setFormatter(file_formatter)
    
    # Create console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(level)
    console_handler.setFormatter(console_formatter)
    
    # Add handlers to logger
    logger.addHandler(file_handler)
    logger.addHandler(rotating_handler)
    logger.addHandler(console_handler)
    
    # Disable propagation to root logger
    logger.propagate = False
    
    # Cache the logger
    loggers[name] = logger
    
    return logger

# Add custom method to log threshold adjustments
def log_threshold_adjustment(logger, old_threshold: float, new_threshold: float, reason: str):
    """
    Log a threshold adjustment with special formatting
    
    Args:
        logger: Logger to use
        old_threshold: Previous threshold value
        new_threshold: New threshold value
        reason: Reason for adjustment
    """
    direction = "INCREASED" if new_threshold > old_threshold else "DECREASED"
    change = abs(new_threshold - old_threshold)
    
    # Custom log with purple color for better visibility
    message = f"THRESHOLD {direction} by {change:.2f} points from {old_threshold:.1f} to {new_threshold:.1f}. Reason: {reason}"
    
    if hasattr(logger, 'handlers') and logger.handlers:
        for handler in logger.handlers:
            if isinstance(handler, logging.StreamHandler) and not isinstance(handler, logging.FileHandler):
                # For console, use purple color
                console_message = f"\033[95mTHRESHOLD {direction}:\033[0m by {change:.2f} points from {old_threshold:.1f} to {new_threshold:.1f}. Reason: {reason}"
                handler.stream.write(console_message + "\n")
                handler.stream.flush()
    
    # Still log normally for files
    logger.warning(message)

# Add method to save threshold adjustment history
def log_adjustment_history(adjustment_history: list, output_dir: Optional[str] = None):
    """
    Save adjustment history to a log file
    
    Args:
        adjustment_history: List of adjustment records
        output_dir: Directory to save the file (uses LOG_DIR if None)
    """
    if output_dir is None:
        output_dir = LOG_DIR
        
    os.makedirs(output_dir, exist_ok=True)
    
    log_file = os.path.join(output_dir, "threshold_adjustments.log")
    
    with open(log_file, 'a') as f:
        for adjustment in adjustment_history:
            timestamp = adjustment.get('timestamp', datetime.now().isoformat())
            old = adjustment.get('old_threshold', 0)
            new = adjustment.get('new_threshold', 0)
            direction = adjustment.get('direction', 'unknown')
            reason = adjustment.get('reason', 'No reason provided')
            
            f.write(f"{timestamp} | {direction.upper()} | {old:.1f} -> {new:.1f} | {reason}\n")

# Main application logger
logger = setup_logger("main")

def log_trade(symbol: str, direction: str, entry_price: float, 
             position_size: float, stop_loss: float = None) -> None:
    """
    Log trade entry information
    
    Args:
        symbol: Trading symbol
        direction: Trade direction (BUY/SELL)
        entry_price: Entry price
        position_size: Position size
        stop_loss: Stop loss price
    """
    trade_log_path = os.path.join(LOG_DIR, "trades.log")
    
    trade_data = {
        "timestamp": datetime.now().isoformat(),
        "symbol": symbol,
        "direction": direction,
        "entry_price": entry_price,
        "position_size": position_size,
        "stop_loss": stop_loss,
        "value": entry_price * position_size
    }
    
    with open(trade_log_path, "a") as trade_log:
        trade_log.write(json.dumps(trade_data) + "\n")
    
    logger.info(f"Trade logged: {direction} {symbol} @ {entry_price}")

def log_performance(metrics: dict) -> None:
    """
    Log performance metrics
    
    Args:
        metrics: Performance metrics dictionary
    """
    performance_log_path = os.path.join(LOG_DIR, "performance.log")
    
    log_data = {
        "timestamp": datetime.now().isoformat(),
        **metrics
    }
    
    with open(performance_log_path, "a") as perf_log:
        perf_log.write(json.dumps(log_data) + "\n")
    
    logger.info(f"Performance logged: {metrics.get('total_trades', 0)} trades, PnL: {metrics.get('total_pnl', 0)}")

def log_error(module: str, message: str, exception: Exception = None) -> None:
    """
    Log an error with detailed information
    
    Args:
        module: Module where error occurred
        message: Error message
        exception: Exception object
    """
    error_logger = setup_logger("errors", logging.ERROR)
    
    if exception:
        error_logger.error(f"{module}: {message} - {str(exception)}")
    else:
        error_logger.error(f"{module}: {message}")
