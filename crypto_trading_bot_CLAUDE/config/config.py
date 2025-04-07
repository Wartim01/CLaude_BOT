"""
Configuration management for the crypto trading bot
"""
import os
import json
import logging
from typing import Dict, Any, Optional
from pathlib import Path

# Define base directory (project root)
BASE_DIR = Path(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Data directories - using exact existing structure
DATA_DIR = os.path.join(BASE_DIR, 'data')
RAW_DATA_DIR = os.path.join(DATA_DIR, 'raw')
PROCESSED_DATA_DIR = os.path.join(DATA_DIR, 'processed')
MARKET_DATA_DIR = os.path.join(DATA_DIR, 'market_data')

# Model directories - using exact existing structure
MODEL_DIR = os.path.join(DATA_DIR, 'models')
MODEL_DATA_DIR = os.path.join(DATA_DIR, 'model_data')
MODEL_CHECKPOINTS_DIR = os.path.join(MODEL_DIR, 'checkpoints')
MODEL_LOGS_DIR = os.path.join(MODEL_DIR, 'logs')
MODEL_ARTIFACTS_DIR = os.path.join(MODEL_DIR, 'artifacts')

# Results and backtest directories - using exact existing structure
BACKTEST_RESULTS_DIR = os.path.join(DATA_DIR, 'backtest-result')
RESULTS_DIR = os.path.join(DATA_DIR, 'results')
VISUALIZATION_DIR = os.path.join(RESULTS_DIR, 'visualizations')

# Define paths
CONFIG_DIR = os.path.join(BASE_DIR, "config")
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
        "timeframes": ["15m", "1h"],
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
    },
    "binance": {
        "production": {
            "API_KEY": "hodtSbBNLSBrDaqAMBEzfdMoGikNynB5wh2cL3xCUVubxMyZYLCP6iRDGffuaCsS",
            "API_SECRET": "pzbN2NalNjWTQOE0aiYTuSWGp44t0fzS7RTH3dsgKTvbmzZNoY6Lam2HAACoTgis"
        },
        "testnet": {
            "API_KEY": "u6cP7KVlRmHLTC4RnGD0jkDZzgEkyK4nXVfIwlxQoM1j9HZZPUu8Vkrbk6ymfIlD",
            "API_SECRET": "P5v5e3Zw24ACZVEnM35NuX3q98ZX29b3tfVHkyzhuEjtvITfCnZUFMKExm8gV2c"
        }
    }
}

# API configuration
USE_TESTNET = False
MAX_API_RETRIES = 3         # Maximum number of API retries
API_RETRY_DELAY = 1         # Delay (in seconds) for retry backoff

# Export the API keys for use in other modules
API_KEYS = {
    "binance": {
        "production": {
            "API_KEY": DEFAULT_CONFIG["binance"]["production"]["API_KEY"],
            "API_SECRET": DEFAULT_CONFIG["binance"]["production"]["API_SECRET"]
        },
        "testnet": {
            "API_KEY": DEFAULT_CONFIG["binance"]["testnet"]["API_KEY"],
            "API_SECRET": DEFAULT_CONFIG["binance"]["testnet"]["API_SECRET"]
        }
    }
}

def load_config(config_path: Optional[str] = None) -> Dict[str, Any]:
    """
    Load configuration from file, create default if not exists
    
    Args:
        config_path: Path to configuration file
        
    Returns:
        Configuration dictionary
    """
    # Use specified config path or default
    if config_path is None:
        config_path = CONFIG_PATH
        
    if not os.path.exists(config_path):
        with open(config_path, 'w') as config_file:
            json.dump(DEFAULT_CONFIG, config_file, indent=4)
        
        print(f"Created default configuration at {config_path}")
        return DEFAULT_CONFIG
    else:
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

def save_config(config: Dict[str, Any], config_path: Optional[str] = None) -> bool:
    """
    Save configuration to file
    
    Args:
        config: Configuration dictionary
        config_path: Path to configuration file
        
    Returns:
        Success status
    """
    if config_path is None:
        config_path = CONFIG_PATH
        
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

# Explicitly export our variables for other modules.
__all__ = [
    "API_KEYS",
    "USE_TESTNET",
    "MAX_API_RETRIES",
    "API_RETRY_DELAY",
    "load_config",
    "save_config",
    "DATA_DIR",
    "MODEL_DIR",
    "BACKTEST_RESULTS_DIR",
    "RESULTS_DIR",
    "LOG_DIR"
]
