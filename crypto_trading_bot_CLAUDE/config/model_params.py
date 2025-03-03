# config/model_params.py
"""
Paramètres de configuration pour les modèles LSTM et autres modèles d'IA
Ces paramètres sont utilisés pour l'architecture, l'entraînement et l'inférence
"""

# Paramètres généraux des modèles
MODEL_VERSION = "1.0.0"
MODEL_CHECKPOINT_DIR = "data/models/checkpoints"
MODEL_PRODUCTION_DIR = "data/models/production"

# Paramètres de base du modèle LSTM
LSTM_BASE_PARAMS = {
    "input_length": 60,           # Longueur de la séquence d'entrée (60 périodes)
    "feature_dim": 30,            # Dimension des caractéristiques (30 caractéristiques principales)
    "lstm_units": [128, 64, 32],  # Unités dans les couches LSTM (3 couches)
    "dropout_rate": 0.3,          # Taux de dropout pour la régularisation
    "learning_rate": 0.001,       # Taux d'apprentissage initial
    "batch_size": 32,             # Taille du batch pour l'entraînement
    "epochs": 100,                # Nombre d'époques maximum
    "patience": 15,               # Patience pour l'early stopping
    "l1_reg": 0.0001,             # Régularisation L1
    "l2_reg": 0.0001,             # Régularisation L2
    "use_attention": True,        # Utiliser des mécanismes d'attention
    "use_residual": True,         # Utiliser des connexions résiduelles
    "prediction_horizons": [12, 24, 96]  # Horizons de prédiction (périodes)
}

# Paramètres des horizons de prédiction
PREDICTION_HORIZONS = [
    # Format: (périodes, nom, est_principal)
    (12, "3h", True),     # Court terme (3h avec timeframe 15min)
    (24, "6h", True),     # Moyen terme (6h avec timeframe 15min)
    (96, "24h", True),    # Long terme (24h avec timeframe 15min)
    (192, "48h", False)   # Très long terme (48h avec timeframe 15min)
]

# Paramètres avancés d'architecture LSTM
LSTM_ADVANCED_PARAMS = {
    "bidirectional": True,         # Utiliser des LSTM bidirectionnels
    "attention_heads": 8,          # Nombre de têtes d'attention
    "attention_size": 64,          # Dimension des mécanismes d'attention
    "dense_units": [64, 32],       # Unités dans les couches denses
    "activation": "relu",          # Fonction d'activation
    "recurrent_activation": "sigmoid",  # Activation récurrente
    "final_activation": {          # Activation de la couche de sortie
        "direction": "sigmoid",    # Pour la direction: sortie binaire
        "volatility": "relu",      # Pour la volatilité: toujours positive
        "momentum": "tanh"         # Pour le momentum: entre -1 et 1
    },
    "use_batch_norm": True,        # Utiliser la normalisation par lots
    "kernel_initializer": "glorot_uniform",  # Initialisation des poids
    "recurrent_initializer": "orthogonal"  # Initialisation récurrente
}

# Paramètres d'entraînement avancés
TRAINING_ADVANCED_PARAMS = {
    "optimizer": "adam",           # Optimiseur (adam, nadam, adamw)
    "loss_functions": {            # Fonctions de perte par facteur
        "direction": "binary_crossentropy",
        "volatility": "mse",
        "momentum": "mse",
        "volume": "mse"
    },
    "loss_weights": {              # Poids des pertes pour chaque facteur
        "direction": 1.0,
        "volatility": 0.7,
        "momentum": 0.7,
        "volume": 0.5
    },
    "lr_schedule": {               # Planification du taux d'apprentissage
        "use_scheduler": True,
        "scheduler_type": "reduce_on_plateau",  # reduce_on_plateau, cyclic, cosine
        "monitor": "val_loss",
        "factor": 0.5,             # Facteur de réduction
        "patience": 5,             # Patience avant réduction
        "min_lr": 1e-6,            # Taux d'apprentissage minimum
        "cooldown": 2              # Période de refroidissement
    },
    "early_stopping": {            # Arrêt précoce
        "monitor": "val_loss",
        "patience": 15,
        "restore_best_weights": True,
        "min_delta": 0.001
    },
    "class_weights": {             # Poids des classes (pour direction)
        0: 1.0,                    # Classe 0 (baisse)
        1: 1.0                     # Classe 1 (hausse)
    },
    "use_sample_weights": False,   # Utiliser des poids d'échantillon
    "shuffle": True,               # Mélanger les données
    "validation_split": 0.15,      # Ratio de validation
    "test_split": 0.15             # Ratio de test
}

# Paramètres de validation croisée temporelle
TIME_SERIES_CV_PARAMS = {
    "n_splits": 5,                # Nombre de plis
    "gap": 12,                    # Écart entre l'entraînement et la validation
    "max_train_size": None,       # Taille maximale de l'entraînement
    "test_size": 48,              # Taille du test (périodes)
    "purge": 6                    # Purge pour éviter la contamination des données
}

# Sélection de caractéristiques pour différentes configurations
FEATURE_SELECTION = {
    "minimal": [
        # Caractéristiques de base essentielles
        "close", "open", "high", "low", "volume",
        "rsi", "bb_width", "ema_9", "ema_21", "atr",
        "macd", "macd_signal", "adx", "plus_di", "minus_di",
        "stoch_k", "stoch_d", "rel_volume_21", "trend_direction"
    ],
    "intermediate": [
        # Inclut minimal plus des caractéristiques supplémentaires
        "body_size_pct", "upper_wick_pct", "lower_wick_pct",
        "momentum_14", "cci", "roc_10", "vwap_diff",
        "obv", "cmf", "bb_percent_b", "choppiness",
        "bullish_pattern_count", "bearish_pattern_count",
        "hour_sin", "hour_cos", "day_sin", "day_cos"
    ],
    "comprehensive": [
        # Toutes les caractéristiques pertinentes
        "price_trend_10_norm", "tsi", "price_acceleration",
        "kc_position", "squeeze_on", "volume_diff_pct",
        "buy_sell_imbalance", "price_efficiency", "up_ratio",
        "bullish_divergence", "bearish_divergence", "macd_bullish_divergence",
        "bull_fractal", "bear_fractal", "hurst_exponent", "lyapunov"
    ],
    "custom": []  # À personnaliser lors de l'entraînement
}

# Paramètres d'architecture avancée pour les modèles
MODEL_ARCHITECTURES = {
    "lstm_simple": {
        "description": "LSTM simple unidirectionnel",
        "lstm_layers": 1,
        "bidirectional": False,
        "use_attention": False,
        "use_residual": False
    },
    "lstm_stacked": {
        "description": "LSTM empilé avec plusieurs couches",
        "lstm_layers": 3,
        "bidirectional": False,
        "use_attention": False,
        "use_residual": True
    },
    "bilstm_attention": {
        "description": "LSTM bidirectionnel avec attention",
        "lstm_layers": 2,
        "bidirectional": True,
        "use_attention": True,
        "use_residual": True
    },
    "hybrid_cnn_lstm": {
        "description": "Hybride CNN-LSTM avec caractéristiques locales et globales",
        "use_cnn": True,
        "cnn_filters": [64, 128],
        "cnn_kernel_sizes": [3, 5],
        "lstm_layers": 2,
        "bidirectional": True,
        "use_attention": True
    },
    "transformer_lstm": {
        "description": "Architecture hybride Transformer-LSTM",
        "use_transformer": True,
        "transformer_layers": 2,
        "transformer_heads": 8,
        "transformer_dim": 64,
        "lstm_layers": 1
    }
}

# Paramètres d'adaptation aux différentes paires de trading
SYMBOL_SPECIFIC_PARAMS = {
    "BTCUSDT": {
        "volatility_multiplier": 1.0,  # Bitcoin est la référence
        "atr_period": 14,
        "optimal_timeframe": "15m",
        "feature_importance": ["rsi", "adx", "bb_width", "macd", "volume"]
    },
    "ETHUSDT": {
        "volatility_multiplier": 1.1,  # Plus volatil que BTC
        "atr_period": 14,
        "optimal_timeframe": "15m",
        "feature_importance": ["rsi", "macd", "stoch_k", "cmf", "adx"]
    },
    "SOLUSDT": {
        "volatility_multiplier": 1.4,  # Beaucoup plus volatil
        "atr_period": 12,
        "optimal_timeframe": "15m",
        "feature_importance": ["rsi", "adx", "macd", "obv", "choppiness"]
    },
    "AVAXUSDT": {
        "volatility_multiplier": 1.3,  # Très volatil
        "atr_period": 12,
        "optimal_timeframe": "15m",
        "feature_importance": ["rsi", "adx", "macd", "volume", "stoch_k"]
    },
    "BNBUSDT": {
        "volatility_multiplier": 1.0,  # Similaire à BTC
        "atr_period": 14,
        "optimal_timeframe": "15m",
        "feature_importance": ["rsi", "macd", "adx", "cci", "cmf"]
    },
    "default": {
        "volatility_multiplier": 1.2,  # Valeur par défaut
        "atr_period": 14,
        "optimal_timeframe": "15m",
        "feature_importance": ["rsi", "adx", "macd", "volume", "bb_width"]
    }
}

# Paramètres d'apprentissage continu
CONTINUOUS_LEARNING_PARAMS = {
    "enabled": True,               # Activer l'apprentissage continu
    "update_frequency": "daily",   # Fréquence des mises à jour (daily, weekly)
    "min_samples": 96,             # Nombre minimum d'échantillons pour la mise à jour
    "max_samples": 10000,          # Nombre maximum d'échantillons à stocker
    "learning_rate_factor": 0.5,   # Facteur de réduction du taux d'apprentissage
    "epochs": 10,                  # Nombre d'époques pour les mises à jour
    "batch_size": 32,              # Taille du batch pour les mises à jour
    "drift_detection": {           # Détection de dérive conceptuelle
        "enabled": True,
        "method": "ks_test",       # ks_test, adwin, ddm
        "threshold": 0.05,         # Seuil de p-valeur
        "window_size": 50,         # Taille de la fenêtre de référence
        "adaptation_rate": 0.3     # Taux d'adaptation après dérive
    },
    "memory_replay": {             # Rejeu de mémoire (éviter l'oubli catastrophique)
        "enabled": True,
        "buffer_size": 5000,       # Taille du buffer
        "sample_ratio": 0.3,       # Ratio d'échantillons à rejouer
        "prioritized": True        # Utiliser l'échantillonnage prioritaire
    }
}

# Paramètres des métriques d'évaluation et de surveillance du modèle
MODEL_EVALUATION_PARAMS = {
    "metrics": [
        "accuracy",                # Précision globale
        "precision",               # Précision (vrais positifs / positifs prédits)
        "recall",                  # Rappel (vrais positifs / positifs réels)
        "f1",                      # Score F1 (moyenne harmonique précision/rappel)
        "roc_auc",                 # Aire sous la courbe ROC
        "mse",                     # Erreur quadratique moyenne (régression)
        "mae",                     # Erreur absolue moyenne (régression)
        "sharpe_ratio",            # Ratio de Sharpe (rendement/risque)
        "sortino_ratio",           # Ratio de Sortino (rendement/risque baissier)
        "calmar_ratio",            # Ratio de Calmar (rendement/drawdown max)
        "directional_accuracy"     # Précision directionnelle (hausse/baisse)
    ],
    "monitoring": {                # Paramètres de surveillance en production
        "performance_threshold": {
            "accuracy": 0.55,      # Seuil minimum de précision
            "sharpe_ratio": 0.5,   # Seuil minimum de ratio de Sharpe
            "max_drawdown": 25.0   # Drawdown maximum toléré (pourcentage)
        },
        "alert_thresholds": {      # Seuils d'alerte
            "accuracy_drop": 0.05, # Alerte si chute de précision
            "profit_factor_min": 1.1,  # Facteur de profit minimum
            "consecutive_losses": 5     # Nombre de pertes consécutives
        },
        "logging_frequency": "daily"  # Fréquence des logs de performance
    }
}

# Caractéristiques sélectionnées pour le modèle final
SELECTED_FEATURES = FEATURE_SELECTION["minimal"] + FEATURE_SELECTION["intermediate"]

# Paramètres optimisés par symbole
def get_symbol_params(symbol: str):
    """
    Récupère les paramètres spécifiques à un symbole
    
    Args:
        symbol: Symbole de la paire de trading
        
    Returns:
        Paramètres spécifiques au symbole
    """
    return SYMBOL_SPECIFIC_PARAMS.get(symbol, SYMBOL_SPECIFIC_PARAMS["default"])

# Paramètres optimisés du modèle LSTM pour le trading agressif
AGGRESSIVE_LSTM_PARAMS = {
    # Fusion des paramètres de base et avancés
    **LSTM_BASE_PARAMS,
    **LSTM_ADVANCED_PARAMS,
    # Paramètres spécifiques pour le trading agressif
    "dropout_rate": 0.25,          # Moins de régularisation pour être plus réactif
    "learning_rate": 0.0015,       # Taux d'apprentissage légèrement plus élevé
    "prediction_horizons": [12, 24, 48],  # Focus sur des horizons plus courts
    "lstm_units": [160, 80, 40],   # Unités plus nombreuses
    "attention_heads": 12,         # Plus de têtes d'attention pour saisir plus de patterns
    "loss_weights": {              # Plus de poids sur la direction et le momentum
        "direction": 1.2,
        "volatility": 0.8,
        "momentum": 1.0,
        "volume": 0.5
    }
}

# Paramètres d'inférence pour les prédictions en temps réel
INFERENCE_PARAMS = {
    "batch_size": 1,               # Taille du batch pour l'inférence
    "confidence_threshold": 0.65,  # Seuil de confiance pour les signaux
    "multi_timeframe": True,       # Utiliser des prédictions multi-timeframes
    "ensemble_method": "weighted", # Méthode d'agrégation des modèles
    "minimum_score": 75,           # Score minimum pour générer un signal
    "throttling": {                # Limitation des prédictions
        "max_signals_per_hour": 6, # Maximum de signaux par heure
        "cooldown_minutes": 15,    # Période de refroidissement entre les signaux
        "consecutive_limit": 3     # Limite de signaux consécutifs
    }
}