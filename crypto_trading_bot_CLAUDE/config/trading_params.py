"""
Paramètres spécifiques à la stratégie de trading
Ces paramètres sont conçus pour être ajustés par l'IA auto-adaptative
"""

# Paramètres de gestion des risques
RISK_PER_TRADE_PERCENT = 3.5  # Pourcentage du capital risqué par trade (5-10%)
MAX_CONCURRENT_TRADES = 3  # Nombre maximum de trades simultanés
MAX_DAILY_TRADES = 25  # Nombre maximum de trades par jour (20-30)
LEVERAGE = 3  # Effet de levier (jusqu'à 5x)

# Paramètres des ordres
STOP_LOSS_PERCENT = 3.3  # Pourcentage de stop-loss (3-5%)
TAKE_PROFIT_PERCENT = 7.5  # Pourcentage de take-profit (5-7%)
TRAILING_STOP_ACTIVATION = 1.2  # Activation du trailing stop après x% de profit
TRAILING_STOP_STEP = 0.5 # Pas du trailing stop
MAX_DRAWDOWN_LIMIT = 30.0     # Ajout d'une limite de drawdown

# Délais et cooldown
MIN_TIME_BETWEEN_TRADES = 4  # Minutes minimum entre trades (3-5 minutes)
MARKET_COOLDOWN_PERIOD = 60  # Minutes de cooldown après détection de marché défavorable

# Paramètres des indicateurs techniques (valeurs par défaut, ajustables par l'IA)
# RSI
RSI_PERIOD = 14
RSI_OVERSOLD = 30  # Seuil de survente
RSI_OVERBOUGHT = 70  # Seuil de surachat

# Bandes de Bollinger
BB_PERIOD = 20
BB_DEVIATION = 2.1

# ATR pour mesure de volatilité
ATR_PERIOD = 14

# EMA pour détection de tendance
EMA_SHORT = 9
EMA_MEDIUM = 21
EMA_LONG = 50

# ADX pour force de tendance
ADX_PERIOD = 14
ADX_THRESHOLD = 25  # Seuil pour considérer une tendance comme forte

# Paramètres du système de scoring
MINIMUM_SCORE_TO_TRADE = 72  # Score minimum pour entrer en position (0-100)

# Facteurs d'adaptation de l'IA
LEARNING_RATE = 0.05  # Taux d'apprentissage pour l'ajustement des paramètres