"""
Logger setup and utilities
"""
import os
import logging
from logging.handlers import RotatingFileHandler
import json
from datetime import datetime

from config.config import LOG_DIR

# Create log directory if it doesn't exist
os.makedirs(LOG_DIR, exist_ok=True)

# Set up logging format
FORMATTER = logging.Formatter(
    "%(asctime)s — %(name)s — %(levelname)s — %(message)s"
)

def setup_logger(name: str, log_file: str = None, level=logging.INFO) -> logging.Logger:
    """
    Set up logger with console and file handlers
    
    Args:
        name: Logger name
        log_file: Log file name (default: <name>.log)
        level: Logging level
        
    Returns:
        Logger instance
    """
    # Use default log file name if not specified
    if log_file is None:
        log_file = f"{name.lower()}.log"
    
    # Create full path to log file
    log_file_path = os.path.join(LOG_DIR, log_file)
    
    # Create logger
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    # Remove existing handlers to avoid duplicates
    if logger.handlers:
        logger.handlers = []
    
    # Create console handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(FORMATTER)
    logger.addHandler(console_handler)
    
    # Create file handler
    file_handler = RotatingFileHandler(
        log_file_path, maxBytes=10485760, backupCount=5
    )
    file_handler.setFormatter(FORMATTER)
    logger.addHandler(file_handler)
    
    return logger

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
    error_logger = setup_logger("errors", "errors.log", logging.ERROR)
    
    if exception:
        error_logger.error(f"{module}: {message} - {str(exception)}")
    else:
        error_logger.error(f"{module}: {message}")
