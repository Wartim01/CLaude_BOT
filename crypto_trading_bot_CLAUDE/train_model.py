#!/usr/bin/env python
"""
Script d’entraînement du modèle de prédiction.
Ce script permet de choisir le type de modèle (LSTM ou Transformer) via un argument de ligne de commande
et d’entraîner le modèle en utilisant les hyperparamètres fournis ou par défaut.
"""

import os
import sys
import argparse
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import json
import tensorflow as tf
import optuna
from data_generator import create_dataset  # Intégration du nouveau script

# Fix Unicode encoding issues on Windows
if sys.platform.startswith('win'):
    # Force UTF-8 encoding for stdout/stderr
    import codecs
    sys.stdout = codecs.getwriter('utf-8')(sys.stdout.buffer, 'strict')
    sys.stderr = codecs.getwriter('utf-8')(sys.stderr.buffer, 'strict')
    # Set console code page to UTF-8
    os.system('chcp 65001')
    # Set environment variable for child processes
    os.environ["PYTHONIOENCODING"] = "utf-8"

from config.config import DATA_DIR, MODEL_CHECKPOINTS_DIR
from utils.logger import setup_logger
from ai.models.lstm_model import EnhancedLSTMModel  # Use advanced model with attention & residuals
from ai.models.feature_engineering import FeatureEngineering

logger = setup_logger("train_model")

def load_best_params(symbol="BTCUSDT", timeframe="15m"):
    """
    Load best parameters from hyperparameter optimization results
    
    Args:
        symbol: Trading pair symbol
        timeframe: Timeframe for the model
        
    Returns:
        Dictionary with best parameters or None if not found
    """
    # Try to find the most recent best_params file
    optimization_dir = os.path.join(DATA_DIR, "models", "optimization")
    if not os.path.exists(optimization_dir):
        logger.warning(f"Optimization directory not found: {optimization_dir}")
        return None
    
    # Get all best_params files
    param_files = [f for f in os.listdir(optimization_dir) if f.startswith("best_params_") and f.endswith(".json")]
    if not param_files:
        logger.warning("No best_params files found in optimization directory")
        return None
    
    # Sort by date (newest first)
    param_files.sort(reverse=True)
    best_params_file = os.path.join(optimization_dir, param_files[0])
    
    try:
        with open(best_params_file, 'r') as f:
            params_data = json.load(f)
            
        logger.info(f"Loaded best parameters from {best_params_file}")
        return params_data.get("best_params", None)
    except Exception as e:
        logger.error(f"Error loading best parameters: {str(e)}")
        return None

# Import model parameters
from config.model_params import LSTM_DEFAULT_PARAMS, LSTM_OPTIMIZED_PARAMS

def parse_args():
    parser = argparse.ArgumentParser(description="Entraînement et évaluation du modèle de prédiction")
    parser.add_argument("--symbol", type=str, required=True, help="Paire de trading (ex: BTCUSDT)")
    parser.add_argument("--timeframe", type=str, required=True, help="Intervalle de temps (ex: 15m, 1h)")
    parser.add_argument("--data_path", type=str, required=True, help="Répertoire des données de marché")
    # Ajout d'un argument pour sélectionner le type de modèle
    parser.add_argument("--model_type", type=str, default="lstm", choices=["lstm", "transformer"],
                        help="Type de modèle à entraîner ('lstm' ou 'transformer')")
    parser.add_argument("--validation_split", type=float, default=0.2, help="Fraction of data to use as validation split")
    # New arguments:
    parser.add_argument("--epochs", type=int, default=100, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size for training")
    parser.add_argument("--lstm_units", type=lambda s: [int(item) for item in s.split(',')], default=[128,64,32],
                        help="Comma-separated list of LSTM units (e.g., 128,64,32)")
    parser.add_argument("--dropout", type=float, default=0.5, help="Dropout rate for training")
    # Add the missing learning rate argument:
    parser.add_argument("--learning_rate", type=float, default=None, help="Learning rate for training")
    # New argument for patience
    parser.add_argument("--patience", type=int, default=10, help="Patience for early stopping and learning rate reduction")
    # Add verbose argument
    parser.add_argument("--verbose", type=int, default=1, choices=[0, 1, 2], 
                        help="Niveau de verbosité pour l'entraînement (0=silencieux, 1=barre de progression, 2=une ligne par époque)")
    parser.add_argument("--sequence_length", type=int, default=60, help="Sequence length for the LSTM input")
    return parser.parse_args()

def load_data(data_path, symbol=None, timeframe=None):
    """
    Charge les données de marché
    
    Args:
        data_path: Chemin vers les données
        symbol: Symbole de la paire de trading (optional)
        timeframe: Intervalle de temps (optional)
        
    Returns:
        DataFrame avec les données
    """
    data_path = os.path.normpath(data_path)  # Normalize the path to handle mixed slashes

    # Added check for data_path existence and fallback to alternative directory if needed
    if not os.path.isdir(data_path):
        alt_data_path = data_path.replace(r"BOT TRADING BIG 2025\data", r"BOT TRADING BIG 2025\crypto_trading_bot_CLAUDE\data")
        if os.path.isdir(alt_data_path):
            logger.info(f"Data directory not found: {data_path}. Using alternative path: {alt_data_path}")
            data_path = alt_data_path
        else:
            raise FileNotFoundError(f"Data directory not found: {data_path}")

    # Déterminer le chemin du fichier
    file_path = os.path.join(data_path, f"{symbol}_{timeframe}.csv") if symbol and timeframe else data_path
    
    # Vérifier si le fichier existe
    if os.path.isdir(file_path) or not os.path.exists(file_path):
        # If it's a directory or the file doesn't exist
        if symbol and timeframe:
            # Chercher un fichier qui contient le symbole et le timeframe
            possible_files = [f for f in os.listdir(data_path) if f.endswith('.csv') and symbol in f and timeframe in f]
            
            if possible_files:
                file_path = os.path.join(data_path, possible_files[0])
                logger.info(f"Utilisation du fichier trouvé: {file_path}")
            else:
                raise FileNotFoundError(f"Aucun fichier de données trouvé pour {symbol}_{timeframe} dans {data_path}")
        else:
            # Handle case when symbol and timeframe are not provided - try to find a CSV file
            possible_files = [f for f in os.listdir(data_path) if f.endswith('.csv')]
            if possible_files:
                file_path = os.path.join(data_path, possible_files[0])
                logger.info(f"Utilisation du premier fichier trouvé: {file_path}")
            else:
                raise FileNotFoundError(f"Aucun fichier CSV trouvé dans {data_path}")
    
    # Charger les données
    logger.info(f"Chargement des données depuis {file_path}...")
    df = pd.read_csv(file_path)
    
    # Vérifier que les colonnes nécessaires existent
    required_columns = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
    
    # Adapter au format si nécessaire
    if 'date' in df.columns and 'timestamp' not in df.columns:
        df['timestamp'] = df['date']
    
    # Vérifier les colonnes
    missing_cols = [col for col in required_columns if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Colonnes manquantes dans les données: {missing_cols}")
    
    # Convertir timestamp en datetime et définir comme index
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df.set_index('timestamp', inplace=True)
    
    # Trier par index pour assurer l'ordre chronologique
    df = df.sort_index()
    
    logger.info(f"Données chargées: {len(df)} lignes de {df.index.min()} à {df.index.max()}")
    
    return df

def train_model(args):
    """
    Entraîne le modèle LSTM avec les paramètres spécifiés
    
    Args:
        args: Arguments de ligne de commande
        
    Returns:
        Le modèle entraîné et les métriques
    """
    # Force the default sequence length to 60 if not provided
    if not hasattr(args, 'sequence_length') or args.sequence_length is None:
        args.sequence_length = 60
    sequence_length = args.sequence_length

    # 1. Charger les données
    df = load_data(args.data_path, args.symbol, args.timeframe)
    
    # 2. Préparer les données
    feature_engineering = FeatureEngineering(save_scalers=True)
    
    # Créer les caractéristiques avancées
    logger.info("Création des caractéristiques...")
    featured_data = feature_engineering.create_features(
        df, 
        include_time_features=True,
        include_price_patterns=True,
        enforce_consistency=True
    )
    
    # Normaliser les données
    logger.info("Normalisation des données...")
    normalized_data = feature_engineering.scale_features(
        featured_data,
        is_training=True,
        method='standard',
        feature_group='lstm'
    )
    
    normalized_data = normalized_data.astype(np.float32)
    
    # Diviser en ensembles d'entraînement et de validation (chronologiquement)
    train_size = int(len(normalized_data) * (1 - args.validation_split))
    train_data = normalized_data.iloc[:train_size]
    val_data = normalized_data.iloc[train_size:]
    
    logger.info(f"Division des données: {len(train_data)} échantillons d'entraînement, {len(val_data)} échantillons de validation")
    
    # Définir la longueur de séquence (utilise l'argument si fourni, sinon la valeur par défaut)
    sequence_length = getattr(args, "sequence_length", None) or LSTM_DEFAULT_PARAMS["sequence_length"]
    
    # Vérifier que les données sont suffisantes pour créer des séquences
    try:
        from config.model_params import PREDICTION_HORIZONS
        if args.timeframe in PREDICTION_HORIZONS:
            horizons = [h[0] for h in PREDICTION_HORIZONS[args.timeframe]]
    except ImportError:
        pass
    if 'horizons' not in locals():
        horizons = [12, 24, 96]

    # Force sequence_length to 60 regardless of what's in args
    args.sequence_length = 60
    sequence_length = 60

    required_samples = sequence_length + max(horizons)
    if len(train_data) < required_samples:
        logger.error(f"Données insuffisantes pour générer des séquences: besoin de {required_samples} échantillons, obtenu {len(train_data)}")
        import sys
        sys.exit(1)
    
    # 3. Créer les séquences et les cibles
    # Utiliser un seul horizon de prédiction (1h)
    if args.timeframe == '15m':
        horizons = [4]  # 4 périodes de 15 minutes = 1 heure
    elif args.timeframe == '1h':
        horizons = [1]  # 1 période de 1 heure = 1 heure
    elif args.timeframe == '5m':
        horizons = [12]  # 12 périodes de 5 minutes = 1 heure
    elif args.timeframe == '4h':
        horizons = [6]  # 6 périodes de 4 heures = 24 heures
    else:
        # Valeur par défaut si le timeframe n'est pas reconnu
        horizons = [1]  # Prédire la période suivante
        
    logger.info(f"Utilisation d'un unique horizon de prédiction: {horizons[0]} périodes")
    
    # Determine which parameters to use based on priority:
    # 1. Command line args
    # 2. Optimized params from config
    # 3. Default params from config
    
    # Check if we have optimized parameters for this timeframe
    optimized_params = None
    if args.timeframe in LSTM_OPTIMIZED_PARAMS and LSTM_OPTIMIZED_PARAMS[args.timeframe]["last_optimized"] is not None:
        optimized_params = LSTM_OPTIMIZED_PARAMS[args.timeframe]
        logger.info(f"Using optimized parameters for {args.timeframe} from config")
        logger.info(f"Last optimization: {optimized_params['last_optimized']} with F1 score: {optimized_params['f1_score']:.4f}")
    
    # Start with default parameters
    params_source = LSTM_DEFAULT_PARAMS
    # Override defaults if not supplied by args
    params_source["lstm_units"] = [136, 68, 34]
    params_source["dropout_rate"] = 0.42
    params_source["learning_rate"] = 0.0001575
    params_source["l1_regularization"] = 0.0008123
    params_source["l2_regularization"] = 0.0003143
    
    # Override with optimized params if available
    if optimized_params:
        params_source = optimized_params
    
    # Override with command line args if provided
    lstm_units = args.lstm_units if hasattr(args, "lstm_units") and args.lstm_units else params_source["lstm_units"]
    dropout_rate = getattr(args, "dropout", None) if hasattr(args, "dropout") and getattr(args, "dropout", None) is not None else params_source["dropout_rate"]
    learning_rate = getattr(args, "learning_rate", None) if hasattr(args, "learning_rate") and getattr(args, "learning_rate", None) is not None else params_source["learning_rate"]
    sequence_length = getattr(args, "sequence_length", None) if hasattr(args, "sequence_length") and getattr(args, "sequence_length", None) is not None else params_source["sequence_length"]  # Par défaut, cela vaudra 90
    l1_reg = getattr(args, "l1_reg", None) if hasattr(args, "l1_reg") and getattr(args, "l1_reg", None) is not None else params_source["l1_regularization"]
    l2_reg = getattr(args, "l2_reg", None) if hasattr(args, "l2_reg") and getattr(args, "l2_reg", None) is not None else params_source["l2_regularization"]
    batch_size = args.batch_size if hasattr(args, "batch_size") and args.batch_size is not None else params_source["batch_size"]
    
    # Log the parameters being used
    logger.info(f"Training with parameters: lstm_units={lstm_units}, dropout={dropout_rate}, "
               f"learning_rate={learning_rate}, sequence_length={sequence_length}, "
               f"batch_size={batch_size}, l1_reg={l1_reg}, l2_reg={l2_reg}")
    
    # Créer les séquences d'entrée et les cibles
    logger.info("Création des séquences pour l'entraînement...")
    X_train, y_train = feature_engineering.create_multi_horizon_data(
        train_data, 
        sequence_length=60,  # Force 60 here
        horizons=horizons,
        is_training=True
    )
    # Vérifier que X_train a bien 3 dimensions
    if X_train.ndim != 3:
        logger.error(f"Dimension inattendue pour X_train : {X_train.shape}. Essai pruned.")
        raise optuna.exceptions.TrialPruned()
    
    logger.info(f"Dimensions de X_train: {X_train.shape} (attendu: (*, {sequence_length}, 80))")
    
    logger.info("Création des séquences pour la validation...")
    X_val, y_val = feature_engineering.create_multi_horizon_data(
        val_data, 
        sequence_length=60,  # Force 60 here
        horizons=horizons,
        is_training=True
    )
    if X_val.ndim != 3:
        logger.error(f"Dimension inattendue pour X_val : {X_val.shape}. Essai pruned.")
        raise optuna.exceptions.TrialPruned()
    
    # Add dimension check - ensure X_train and X_val are 3D
    logger.info(f"Vérification des dimensions: X_train = {X_train.shape}, X_val = {X_val.shape}")
    
    if len(X_train.shape) != 3:
        raise ValueError(f"Les données d'entraînement doivent avoir 3 dimensions (trouvé: {len(X_train.shape)})")
    
    if len(X_val.shape) != 3:
        raise ValueError(f"Les données de validation doivent avoir 3 dimensions (trouvé: {len(X_val.shape)})")
    
    # 5. Créer et compiler le modèle with parameters from config/args
    logger.info(f"Création du modèle LSTM avec unités: {lstm_units}")
    
    # Process a small batch of data to determine the actual feature dimension
    logger.info("Determining actual feature dimension from processed data...")
    X_sample, _ = feature_engineering.create_multi_horizon_data(
        normalized_data.iloc[:100],  # Use a small sample
        sequence_length=sequence_length,
        horizons=horizons,
        is_training=True
    )
    
    # Get the actual feature dimension from the processed data
    actual_feature_dim = X_sample.shape[2]
    logger.info(f"Detected actual feature dimension: {actual_feature_dim}")
    
    # Create model with the actual feature dimension
    model = EnhancedLSTMModel(
        input_length=60,
        feature_dim=actual_feature_dim,  # or appropriate value from configuration
        lstm_units=[136, 68, 68],
        dropout_rate=0.4195981825434215,
        learning_rate=0.00015751320499779721,
        use_attention=True,
        use_residual=True,
        prediction_horizons=[(4, "1h", True)]
    )
    
    # 6. Préparer les callbacks
    callbacks = []
    
    # Créer le répertoire pour les checkpoints si nécessaire
    os.makedirs(MODEL_CHECKPOINTS_DIR, exist_ok=True)
    
    # Chemin pour sauvegarder le modèle
    if hasattr(args, "output"):
        model_path = args.output
    else:
        model_path = os.path.join(DATA_DIR, "models", f"lstm_{args.symbol}_{args.timeframe}.keras")
    
    # Créer le répertoire parent si nécessaire
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    
    # Model checkpoint - UPDATED to use .keras extension
    checkpoint_path = os.path.join(MODEL_CHECKPOINTS_DIR, f"lstm_{args.symbol}_{args.timeframe}_checkpoint.keras")
    from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau # type: ignore
    
    callbacks.append(
        ModelCheckpoint(
            filepath=checkpoint_path,
            monitor='val_loss',
            save_best_only=True,
            save_weights_only=False,
            verbose=1
        )
    )
    
    # Early stopping
    if getattr(args, "use_early_stopping", False):
        callbacks.append(
            EarlyStopping(
                monitor='val_loss',
                patience=getattr(args, "patience", 10),
                restore_best_weights=True,
                verbose=1
            )
        )
    
    # Learning rate reduction
    callbacks.append(
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=args.patience // 2,
            min_lr=1e-6,
            verbose=1
        )
    )
    
    # Use the verbose parameter from args instead of hardcoding it
    verbose_level = args.verbose if hasattr(args, "verbose") else 1
    
    # 7. Entraîner le modèle with batch_size from params
    logger.info(f"Début de l'entraînement pour {args.epochs} époques avec batch_size={batch_size}...")
    
    # Ajouter un callback personnalisé pour afficher les performances à chaque époque
    from tensorflow.keras.callbacks import Callback # type: ignore
    class DetailedEpochLogger(Callback):
        def on_epoch_end(self, epoch, logs=None):
            logs = logs or {}
            metrics_str = ' - '.join([f"{k}: {v:.4f}" for k, v in logs.items()])
            # Replace print with logger.info() to ensure metrics appear in unified logs
            logger.info(f"Époque {epoch+1}/{self.params['epochs']} - {metrics_str}")
    
    # Only add the custom logger if verbose level is not already 2
    if verbose_level != 2:
        # Ajouter notre callback à la liste existante
        callbacks.append(DetailedEpochLogger())
    
    # Use the verbose level from arguments
    history = model.model.fit(
        x=X_train,
        y=y_train,
        epochs=args.epochs,
        batch_size=batch_size,
        validation_data=(X_val, y_val),
        callbacks=callbacks,
        verbose=verbose_level  # Use the command line argument value
    )
    
    # 8. Sauvegarder le modèle
    logger.info(f"Sauvegarde du modèle dans {model_path}...")
    # Modify model_path assignment to use default if args.output is None
    if getattr(args, "output", None):
        model_path = args.output
    else:
        model_path = os.path.join(DATA_DIR, "models", f"lstm_{args.symbol}_{args.timeframe}.keras")
    model.save(model_path)
    
    # 9. Sauvegarder l'historique d'entraînement
    history_path = os.path.join(os.path.dirname(model_path), f"history_{args.symbol}_{args.timeframe}.json")
    with open(history_path, 'w') as f:
        # Convertir les valeurs numpy en float pour JSON
        history_dict = {}
        for key, values in history.history.items():
            history_dict[key] = [float(v) for v in values]
        
        json.dump(history_dict, f, indent=2)
    
    # 10. Évaluation du modèle
    if getattr(args, "evaluate_after", False):
        logger.info("Évaluation du modèle sur les données de validation...")
        evaluate_model(model, X_val, y_val, args.symbol, args.timeframe, model_path)
    
    return model, history

def evaluate_model(model, X_val, y_val, symbol, timeframe, model_path):
    """
    Évalue le modèle sur les données de validation
    
    Args:
        model: Modèle LSTM entraîné
        X_val: Données de validation (features)
        y_val: Données de validation (cibles)
        symbol: Symbole de la paire de trading
        timeframe: Intervalle de temps
        model_path: Chemin du modèle sauvegardé
    """
    # Évaluer le modèle
    evaluation = model.model.evaluate(X_val, y_val, verbose=1)
    
    # Générer des prédictions
    predictions = model.model.predict(X_val)
    
    # Avec un seul horizon, pas besoin de boucle ou d'index
    # Les prédictions concernent directement l'horizon choisi (1h)
    y_true = y_val.flatten() if isinstance(y_val, np.ndarray) else y_val[0].flatten()
    y_pred = (predictions.flatten() > 0.5).astype(int) if isinstance(predictions, np.ndarray) else (predictions[0].flatten() > 0.5).astype(int)
    
    # Calculer la précision
    accuracy = np.mean(y_true == y_pred)
    direction_accuracy = {"horizon_1h": accuracy}
    
    # Sauvegarder les métriques d'évaluation
    metrics_path = os.path.join(os.path.dirname(model_path), f"metrics_{symbol}_{timeframe}.json")
    
    metrics = {
        "evaluation": [float(e) for e in evaluation],
        "direction_accuracy": direction_accuracy,
        "timestamp": datetime.now().isoformat()
    }
    
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    
    logger.info(f"Précision de direction: {direction_accuracy}")
    logger.info(f"Métriques sauvegardées dans {metrics_path}")
    
    # Créer des visualisations
    create_evaluation_plots(model, X_val, y_val, predictions, symbol, timeframe, model_path)

def create_evaluation_plots(model, X_val, y_val, predictions, symbol, timeframe, model_path):
    """
    Crée des visualisations des performances du modèle
    
    Args:
        model: Modèle LSTM entraîné
        X_val: Données de validation (features)
        y_val: Données de validation (cibles)
        predictions: Prédictions du modèle
        symbol: Symbole de la paire de trading
        timeframe: Intervalle de temps
        model_path: Chemin du modèle sauvegardé
    """
    # Créer le répertoire pour les visualisations
    viz_dir = os.path.join(os.path.dirname(model_path), "visualizations")
    os.makedirs(viz_dir, exist_ok=True)
    
    import matplotlib.pyplot as plt
    import seaborn as sns
    from sklearn.metrics import confusion_matrix
    
    # Handle different data structures - y_val could be a single array or a list of arrays
    # Convert both y_val and predictions to consistent formats
    if isinstance(predictions, list) and len(predictions) > 0:
        # Multiple output model (multi-horizon)
        num_outputs = len(predictions)
    else:
        # Single output model or single horizon
        num_outputs = 1
        if not isinstance(predictions, list):
            predictions = [predictions]  # Wrap in list for consistent handling
        if not isinstance(y_val, list):
            y_val = [y_val]  # Wrap in list for consistent handling
            
    # Ensure we don't exceed array bounds
    num_outputs = min(num_outputs, len(y_val)) if isinstance(y_val, list) else 1
    
    if num_outputs == 0:
        logger.warning("No outputs to evaluate. Skipping visualization generation.")
        return
        
    # 1. Direction prediction accuracy per horizon
    plt.figure(figsize=(10, 6))
    horizons = []
    accuracies = []
    
    # Safely iterate through outputs
    for i in range(num_outputs):
        if i < len(predictions) and (isinstance(y_val, list) and i < len(y_val)):
            # Extract true values - handle both single array and list of arrays
            if isinstance(y_val, list):
                y_true = y_val[i].flatten() if hasattr(y_val[i], 'flatten') else np.array(y_val[i]).flatten()
            else:
                y_true = y_val.flatten() if hasattr(y_val, 'flatten') else np.array(y_val).flatten()
                
            # Extract predictions and convert to binary class
            if isinstance(predictions[i], np.ndarray):
                y_pred = (predictions[i].flatten() > 0.5).astype(int)
            else:
                # Handle other possible formats (e.g., list)
                y_pred = (np.array(predictions[i]).flatten() > 0.5).astype(int)
                
            # Calculate accuracy
            accuracy = np.mean(y_true == y_pred) * 100
            
            horizons.append(f"H{i+1}")
            accuracies.append(accuracy)
    
    # Create bar chart if we have data
    if horizons and accuracies:
        plt.bar(horizons, accuracies, color='skyblue')
        plt.axhline(y=50, color='r', linestyle='--', alpha=0.5)  # 50% line (random)
        plt.ylim([0, 100])
        plt.title(f'Précision de Prédiction de Direction par Horizon - {symbol} {timeframe}')
        plt.ylabel('Précision (%)')
        plt.xlabel('Horizon')
        
        for i, v in enumerate(accuracies):
            plt.text(i, v + 1, f"{v:.1f}%", ha='center')
        
        plt.savefig(os.path.join(viz_dir, f"{symbol}_{timeframe}_direction_accuracy.png"))
        plt.close()
    
    # 2. Confusion matrix for the first horizon - only create if we have at least one output
    if num_outputs > 0:
        try:
            # Get true values and predictions for the first horizon
            if isinstance(y_val, list):
                y_true = y_val[0].flatten() if hasattr(y_val[0], 'flatten') else np.array(y_val[0]).flatten()
            else:
                y_true = y_val.flatten() if hasattr(y_val, 'flatten') else np.array(y_val).flatten()
                
            if isinstance(predictions[0], np.ndarray):
                y_pred = (predictions[0].flatten() > 0.5).astype(int)
            else:
                y_pred = (np.array(predictions[0]).flatten() > 0.5).astype(int)
                
            # Create confusion matrix
            cm = confusion_matrix(y_true, y_pred)
            
            plt.figure(figsize=(8, 6))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
            plt.title(f'Matrice de Confusion - {symbol} {timeframe}')
            plt.ylabel('Vrai')
            plt.xlabel('Prédit')
            plt.savefig(os.path.join(viz_dir, f"{symbol}_{timeframe}_confusion_matrix.png"))
            plt.close()
        except Exception as e:
            logger.warning(f"Error creating confusion matrix: {str(e)}")
    
    logger.info(f"Visualisations sauvegardées dans {viz_dir}")

def main():
    args = parse_args()
    train_model(args)

if __name__ == "__main__":
    main()

