# config/model_params.py
"""
Configuration parameters for various model architectures
This file centralizes hyperparameters for different models to ensure consistency
"""

# LSTM model parameters
LSTM_DEFAULT_PARAMS = {
    "lstm_units": [136, 68, 68],
    "dropout_rate": 0.39279757672456206,
    "learning_rate": 0.0006251373574521745,
    "batch_size": 64,
    "sequence_length": 30,
    "l1_regularization": 0.0003967605077052988,
    "l2_regularization": 6.358358856676247e-05,
    # Other parameters preserved
    "use_attention": True,
    "use_residual": True,
    "epochs": 100,
    "early_stopping_patience": 15,
    "reduce_lr_patience": 7
}

# [AUTO-UPDATE SECTION] - DO NOT EDIT MANUALLY
# This section is automatically updated by hyperparameter_search.py
# Last optimization: 2025-03-15 18:39:30 (timeframe: 15m, F1: 0.0000)
LSTM_OPTIMIZED_PARAMS = {
    "15m": {
        "lstm_units": [136, 68, 68],
        "dropout_rate": 0.39279757672456206,
        "learning_rate": 0.0006251373574521745,
        "batch_size": 64,
        "sequence_length": 30,
        "l1_regularization": 0.0003967605077052988,
        "l2_regularization": 6.358358856676247e-05,
        "use_attention": True,
        "use_residual": True,
        "last_optimized": "2025-03-15 18:39:30",
        "f1_score": 0.0
    },
}

# Transformer model parameters
TRANSFORMER_DEFAULT_PARAMS = {
    "num_layers": 4,
    "num_heads": 8,
    "head_dim": 32,
    "ff_dim": 128,
    "dropout_rate": 0.1,
    "learning_rate": 0.0001,
    "batch_size": 64,
    "epochs": 100,
    "sequence_length": 60,
    "early_stopping_patience": 10,
    "reduce_lr_patience": 5
}

# CNN-LSTM hybrid model parameters
CNN_LSTM_DEFAULT_PARAMS = {
    "lstm_units": [136, 68, 68],
    "dropout_rate": 0.39279757672456206,
    "learning_rate": 0.0006251373574521745,
    "batch_size": 64,
    "sequence_length": 30,
    "l1_regularization": 0.0003967605077052988,
    "l2_regularization": 6.358358856676247e-05,
    # Other parameters preserved
    "use_attention": True,
    "use_residual": True,
    "epochs": 100,
    "early_stopping_patience": 15,
    "reduce_lr_patience": 7
}

# Feature selection settings 
FEATURE_GROUPS = {
    "basic": [
        "close", "volume", "rsi", "macd", "bb_width", "atr"
    ],
    "technical": [
        "close", "volume", "rsi", "macd", "macd_signal", "macd_hist",
        "bb_upper", "bb_lower", "bb_width", "bb_percent_b",
        "ema_9", "ema_21", "ema_50", "ema_200",
        "atr", "adx", "stoch_k", "stoch_d",
        "mom_10", "obv", "dist_ma_50"
    ],
    "advanced": [
        # Price action features
        "close", "return_1", "return_3", "return_5", "return_10", 
        "body_size_percent", "upper_wick_percent", "lower_wick_percent",
        
        # Technical indicators
        "rsi", "macd", "macd_signal", "macd_hist", "adx", "plus_di", "minus_di",
        "bb_width", "bb_percent_b", "atr", "atr_percent", "stoch_k", "stoch_d",
        
        # Price relative to moving averages
        "dist_ma_20", "dist_ma_50", "dist_ma_200",
        
        # Volume metrics
        "volume", "rel_volume_5", "rel_volume_10", "obv",
        
        # Volatility measures
        "volatility_5", "volatility_10", "volatility_ratio",
        
        # Statistical features
        "return_zscore_10", "fractal_dimension", "hurst_exponent"
    ],
    "full": "all"  # Use all available features
}

# Training horizons for different timeframes - Modified for single horizon focus
PREDICTION_HORIZONS = {
    # Format: (periods, name, is_main)
    "1m": [(4, "1h", True)],  # 1 hour ahead only
    "5m": [(12, "1h", True)], # 1 hour ahead only
    "15m": [(4, "1h", True)], # 1 hour ahead only (4 periods of 15min = 1h)
    "1h": [(1, "1h", True)],  # 1 hour ahead only
    "4h": [(6, "24h", True)], # 24 hours ahead only
    "1d": [(7, "7d", True)]   # 7 days ahead only
}

# Model optimization settings
OPTIMIZATION_PARAMS = {
    "n_trials": 30,
    "timeout": 6 * 3600,  # 6 hours in seconds
    "pruning": True,
    "study_direction": "maximize",  # "minimize" for loss, "maximize" for metrics like accuracy or f1
    "metric": "f1_score",  # Metric to optimize: "loss", "accuracy", "f1_score"
    "cross_validation_folds": 3
}

# Dataset preparation parameters
DATASET_PARAMS = {
    "validation_split": 0.15,
    "test_split": 0.15,
    "max_samples": 50000,  # Maximum samples to use (None for all)
    "sample_weights_enabled": True,  # Whether to use sample weights for imbalanced data
    "negative_sample_weight": 1.25  # Weight for negative samples (for balancing)
}

# Ensemble model parameters
ENSEMBLE_PARAMS = {
    "models": ["lstm", "transformer"],  # Models to include in ensemble
    "voting": "soft",  # "hard" or "soft"
    "weights": [0.7, 0.3]  # Weights for each model's predictions
}