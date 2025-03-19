#!/usr/bin/env python
"""
Script d’entraînement du modèle de prédiction.
Ce script permet de choisir le type de modèle (LSTM ou Transformer) via un argument de ligne de commande
et d’entraîner le modèle en utilisant les hyperparamètres fournis ou par défaut.
"""

import os
import argparse
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import json
import tensorflow as tf

from config.config import DATA_DIR, MODEL_CHECKPOINTS_DIR
from utils.logger import setup_logger
from ai.models.lstm_model import LSTMModel
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
    return parser.parse_args()

def load_data(data_path, symbol, timeframe):
    """
    Charge les données de marché
    
    Args:
        data_path: Chemin vers les données
        symbol: Symbole de la paire de trading
        timeframe: Intervalle de temps
        
    Returns:
        DataFrame avec les données
    """
    # Déterminer le chemin du fichier
    file_path = os.path.join(data_path, f"{symbol}_{timeframe}.csv")
    
    # Vérifier si le fichier existe
    if not os.path.exists(file_path):
        # Chercher un fichier qui contient le symbole et le timeframe
        possible_files = [f for f in os.listdir(data_path) if f.endswith('.csv') and symbol in f and timeframe in f]
        
        if possible_files:
            file_path = os.path.join(data_path, possible_files[0])
            logger.info(f"Utilisation du fichier trouvé: {file_path}")
        else:
            raise FileNotFoundError(f"Aucun fichier de données trouvé pour {symbol}_{timeframe} dans {data_path}")
    
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
    # 1. Charger les données
    df = load_data(args.data_path, args.symbol, args.timeframe)
    
    # 2. Préparer les données
    feature_engineering = FeatureEngineering(save_scalers=True)
    
    # Créer les caractéristiques avancées
    logger.info("Création des caractéristiques...")
    featured_data = feature_engineering.create_features(
        df, 
        include_time_features=True,
        include_price_patterns=True
    )
    
    # Normaliser les données
    logger.info("Normalisation des données...")
    normalized_data = feature_engineering.scale_features(
        featured_data,
        is_training=True,
        method='standard',
        feature_group='lstm'
    )
    
    # Diviser en ensembles d'entraînement et de validation (chronologiquement)
    train_size = int(len(normalized_data) * (1 - args.validation_split))
    train_data = normalized_data.iloc[:train_size]
    val_data = normalized_data.iloc[train_size:]
    
    logger.info(f"Division des données: {len(train_data)} échantillons d'entraînement, {len(val_data)} échantillons de validation")
    
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
    
    # Override with optimized params if available
    if optimized_params:
        params_source = optimized_params
    
    # Override with command line args if provided
    lstm_units = args.lstm_units if args.lstm_units else params_source["lstm_units"]
    dropout_rate = getattr(args, "dropout", None) if getattr(args, "dropout", None) is not None else params_source["dropout_rate"]
    learning_rate = getattr(args, "learning_rate", None) if getattr(args, "learning_rate", None) is not None else params_source["learning_rate"]
    sequence_length = getattr(args, "sequence_length", None) if getattr(args, "sequence_length", None) is not None else params_source["sequence_length"]
    l1_reg = getattr(args, "l1_reg", None) if getattr(args, "l1_reg", None) is not None else params_source["l1_regularization"]
    l2_reg = getattr(args, "l2_reg", None) if getattr(args, "l2_reg", None) is not None else params_source["l2_regularization"]
    batch_size = args.batch_size if args.batch_size is not None else params_source["batch_size"]
    
    # Log the parameters being used
    logger.info(f"Training with parameters: lstm_units={lstm_units}, dropout={dropout_rate}, "
               f"learning_rate={learning_rate}, sequence_length={sequence_length}, "
               f"batch_size={batch_size}, l1_reg={l1_reg}, l2_reg={l2_reg}")
    
    # Créer les séquences d'entrée et les cibles
    logger.info("Création des séquences pour l'entraînement...")
    X_train, y_train = feature_engineering.create_multi_horizon_data(
        train_data, 
        sequence_length=sequence_length,  # Use the sequence length from params
        horizons=horizons,
        is_training=True
    )
    logger.info(f"Dimensions de X_train: {X_train.shape} (attendu: (*, {sequence_length}, 80))")
    
    logger.info("Création des séquences pour la validation...")
    X_val, y_val = feature_engineering.create_multi_horizon_data(
        val_data, 
        sequence_length=sequence_length,  # Use the sequence length from params here too
        horizons=horizons,
        is_training=True
    )
    
    # Add dimension check - ensure X_train and X_val are 3D
    logger.info(f"Vérification des dimensions: X_train = {X_train.shape}, X_val = {X_val.shape}")
    
    if len(X_train.shape) != 3:
        raise ValueError(f"Les données d'entraînement doivent avoir 3 dimensions (trouvé: {len(X_train.shape)})")
    
    if len(X_val.shape) != 3:
        raise ValueError(f"Les données de validation doivent avoir 3 dimensions (trouvé: {len(X_val.shape)})")
    
    # 5. Créer et compiler le modèle with parameters from config/args
    logger.info(f"Création du modèle LSTM avec unités: {lstm_units}")
    model = LSTMModel(
        input_length=sequence_length,  # Use the sequence length from params
        feature_dim=X_train.shape[2],
        lstm_units=lstm_units,
        dropout_rate=dropout_rate,
        learning_rate=learning_rate,
        use_attention=getattr(args, "use_attention", False),
        prediction_horizons=horizons,
        l1_reg=l1_reg,
        l2_reg=l2_reg
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
    from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
    
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
    
    # 7. Entraîner le modèle with batch_size from params
    logger.info(f"Début de l'entraînement pour {args.epochs} époques avec batch_size={batch_size}...")
    history = model.model.fit(
        x=X_train,
        y=y_train,
        epochs=args.epochs,
        batch_size=batch_size,
        validation_data=(X_val, y_val),
        callbacks=callbacks,
        verbose=getattr(args, "verbose", 1)
    )
    
    # 8. Sauvegarder le modèle
    logger.info(f"Sauvegarde du modèle dans {model_path}...")
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
    """Point d'entrée principal"""
    args = parse_args()
    
    try:
        # Entraîner le modèle
        model, history = train_model(args)
        
        logger.info("Entraînement terminé avec succès!")
        
    except Exception as e:
        logger.error(f"Erreur lors de l'entraînement: {str(e)}")
        import traceback
        traceback.print_exc()
        import sys
        sys.exit(1)

if __name__ == "__main__":
    main()

def create_model(input_shape, lstm_units):
    """
    Creates a model with the specified input shape and LSTM units
    
    Args:
        input_shape: Shape of input data
        lstm_units: Number of units in LSTM layers
        
    Returns:
        Compiled Keras model
    """
    from tensorflow.keras.layers import Input, Bidirectional, LSTM, Dropout, Dense
    from tensorflow.keras.models import Model
    from tensorflow.keras.optimizers import Adam
    # Start with input shape
    inputs = Input(shape=input_shape)
    x = inputs
    
    # Add LSTM layers
    for i, units in enumerate(lstm_units):
        return_sequences = i < len(lstm_units) - 1
        x = Bidirectional(LSTM(units, return_sequences=return_sequences))(x)
        x = Dropout(0.3)(x)
    
    # Dense layers
    x = Dense(64, activation='relu')(x)
    x = Dropout(0.2)(x)
    
    # Output layers
    outputs = []
    for horizon in range(3):  # for 3 different prediction horizons
        name = f'direction_{horizon+1}'
        outputs.append(Dense(1, activation='sigmoid', name=name)(x))
    
    # Create the model
    model = Model(inputs=inputs, outputs=outputs)
    
    # Compile with proper loss and metrics
    loss_functions = ['binary_crossentropy'] * len(outputs)
    
    # Create optimizer
    optimizer = Adam(learning_rate=0.001)
    
    # Fix the metrics format - use a simple string metric for all outputs
    model.compile(
        optimizer=optimizer,
        loss=loss_functions,
        metrics='accuracy'  # Apply accuracy to all outputs
    )
    
    return model

