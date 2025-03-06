"""
Configuration globale du bot de trading crypto
"""
import os
from dotenv import load_dotenv
import logging

# Chargement des variables d'environnement
load_dotenv()

# Configuration de l'API Binance
BINANCE_API_KEY = os.getenv("BINANCE_API_KEY", "")
BINANCE_API_SECRET = os.getenv("BINANCE_API_SECRET", "")
USE_TESTNET = os.getenv("USE_TESTNET", "True").lower() in ("true", "1", "t")

# Paramètres généraux
INITIAL_CAPITAL = 200  # USDT
BASE_CURRENCY = "USDT"
TRADING_PAIRS = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "AVAXUSDT", "BNBUSDT"]
DEFAULT_TRADING_PAIR = "BTCUSDT"

# Intervalles de temps pour l'analyse
PRIMARY_TIMEFRAME = "15m"  # Timeframe principal (15 minutes)
SECONDARY_TIMEFRAMES = ["1h", "4h"]  # Timeframes secondaires pour confirmation

# Chemins des répertoires
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
LOG_DIR = os.path.join(BASE_DIR, "logs")

# Création des répertoires s'ils n'existent pas
for directory in [DATA_DIR, LOG_DIR]:
    if not os.path.exists(directory):
        os.makedirs(directory)

# Configuration du logging
LOG_LEVEL = logging.INFO
LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
LOG_FILE = os.path.join(LOG_DIR, "trading_bot.log")

# Paramètres du système
MAX_API_RETRIES = 3
API_RETRY_DELAY = 2  # secondes

# Paramètres de notification (à implémenter ultérieurement)
ENABLE_NOTIFICATIONS = False
NOTIFICATION_EMAIL = os.getenv("NOTIFICATION_EMAIL", "")



MODEL_CHECKPOINTS_DIR = "C:\\Users\\timot\\OneDrive\\Bureau\\BOT TRADING BIG 2025\\crypto_trading_bot_CLAUDE\\data\\models\\checkpoints"
