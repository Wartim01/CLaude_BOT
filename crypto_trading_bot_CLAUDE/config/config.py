"""
Configuration globale du bot de trading crypto
"""
import os
from dotenv import load_dotenv
import logging

# Chargement des variables d'environnement
load_dotenv()

# Configuration de l'API Binance
# Structure correcte des clés API
API_KEYS = {
    "BINANCE": {
        # Production keys
        "key": "hodtSbBNLSBrDaqAMBEzfdMoGikNynB5wh2cL3xCUVubxMyZYLCP6iRDGffuaCsS",
        "secret": "pzbN2NalNjWTQOE0aiYTuSWGp44t0fzS7RTH3dsgKTvbmzZNoY6Lam2HAACoTgis"
    },
    "BINANCE_TESTNET": {
        # Testnet keys
        "key": "u6cP7KVlRmHLTC4RnGD0jkDZzgEkyK4nXVfIwlxQoM1j9HZZPUu8Vkrbk6ymfIlD",
        "secret": "P5v5e3Zw24ACZVEnM35NuX3q98ZX29b3tfVHkyzhuEjtvITfCnZUFMKExm8gV2c"
    }
}

# Utiliser testnet ou production
USE_TESTNET = os.getenv("USE_TESTNET", "True").lower() in ("true", "1", "t")

# Récupérer les bonnes clés en fonction du mode
if USE_TESTNET:
    ACTIVE_API_KEY = API_KEYS["BINANCE_TESTNET"]["key"]
    ACTIVE_API_SECRET = API_KEYS["BINANCE_TESTNET"]["secret"]
else:
    ACTIVE_API_KEY = API_KEYS["BINANCE"]["key"]
    ACTIVE_API_SECRET = API_KEYS["BINANCE"]["secret"]

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

MODEL_CHECKPOINTS_DIR = os.path.join(DATA_DIR, "models", "checkpoints")
