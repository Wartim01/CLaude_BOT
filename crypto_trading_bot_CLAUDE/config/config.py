"""
Configuration management for the crypto trading bot
"""
import os
import json
import logging
from typing import Dict, Any, Optional

# Define paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CONFIG_DIR = os.path.join(BASE_DIR, "config")
DATA_DIR = os.path.join(BASE_DIR, "data")
LOG_DIR = os.path.join(BASE_DIR, "logs")
CONFIG_PATH = os.path.join(CONFIG_DIR, "config.json")

# Default configuration
DEFAULT_CONFIG = {
    "bot": {
        "name": "CryptoTradingBot",
        "version": "1.0.0",
        "check_interval_seconds": 60,
        "log_level": "INFO"
    },
    "exchange": {
        "name": "paper",  # "binance", "paper"
        "api_key": "",
        "api_secret": "",
        "testnet": True
    },
    "trading": {
        "mode": "paper",  # "live", "paper", "backtest"
        "pairs": ["BTCUSDT", "ETHUSDT", "BNBUSDT", "ADAUSDT", "XRPUSDT"],
        "timeframes": ["1h", "4h", "1d"],
        "risk_per_trade": 0.02,  # 2% risk per trade
        "max_risk": 0.10,        # 10% maximum risk across all trades
        "auto_select_pairs": False
    },
    "strategies": {
        "active": ["trend_following", "breakout"],
        "weights": {
            "trend_following": 0.6,
            "breakout": 0.4
        },
        "parameters": {
            "trend_following": {
                "ema_short": 9,
                "ema_long": 21,
                "atr_period": 14
            },
            "breakout": {
                "lookback_period": 20,
                "volatility_factor": 1.5
            }
        }
    },
    "risk_management": {
        "stop_loss": True,
        "trailing_stop": True,
        "take_profit": True,
        "trailing_stop_distance": 0.02,  # 2% trailing stop
        "take_profit_targets": [0.03, 0.05, 0.1]  # 3%, 5%, 10% targets
    },
    "notifications": {
        "enabled": False,
        "telegram_token": "",
        "telegram_chat_id": "",
        "notify_on": ["trade_opened", "trade_closed", "bot_started", "bot_stopped", "error"]
    }
}

# Create directories if they don't exist
for directory in [CONFIG_DIR, DATA_DIR, LOG_DIR]:
    os.makedirs(directory, exist_ok=True)

def load_config(config_path: Optional[str] = CONFIG_PATH) -> Dict[str, Any]:
    """
    Load configuration from file, create default if not exists
    
    Args:
        config_path: Path to configuration file
        
    Returns:
        Configuration dictionary
    """
    # Use specified config path or default
    config_path = config_path or CONFIG_PATH
    
    # Create default config if file doesn't exist
    if not os.path.exists(config_path):
        with open(config_path, 'w') as config_file:
            json.dump(DEFAULT_CONFIG, config_file, indent=4)
        
        print(f"Created default configuration at {config_path}")
        return DEFAULT_CONFIG
    
    # Load existing config
    try:
        with open(config_path, 'r') as config_file:
            config = json.load(config_file)
        
        # Ensure all default settings exist (for backwards compatibility)
        merged_config = DEFAULT_CONFIG.copy()
        _update_dict_recursively(merged_config, config)
        
        return merged_config
    
    except Exception as e:
        print(f"Error loading config: {str(e)}. Using default configuration.")
        return DEFAULT_CONFIG

def save_config(config: Dict[str, Any], config_path: Optional[str] = CONFIG_PATH) -> bool:
    """
    Save configuration to file
    
    Args:
        config: Configuration dictionary
        config_path: Path to configuration file
        
    Returns:
        Success status
    """
    try:
        with open(config_path, 'w') as config_file:
            json.dump(config, config_file, indent=4)
        return True
    except Exception as e:
        print(f"Error saving config: {str(e)}")
        return False

def _update_dict_recursively(target_dict: Dict, source_dict: Dict) -> None:
    """
    Update target dictionary with values from source dictionary recursively
    
    Args:
        target_dict: Target dictionary to update
        source_dict: Source dictionary with values to copy
    """
    for key, value in source_dict.items():
        if key in target_dict and isinstance(target_dict[key], dict) and isinstance(value, dict):
            _update_dict_recursively(target_dict[key], value)
        else:
            target_dict[key] = value
