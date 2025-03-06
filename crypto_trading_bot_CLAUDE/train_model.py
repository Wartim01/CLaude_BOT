#!/usr/bin/env python
"""
Script d'entraînement du modèle LSTM pour la prédiction des mouvements de marché
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

def parse_args():
    """Parse les arguments de ligne de commande"""
    parser = argparse.ArgumentParser(description="Entraînement et évaluation du modèle LSTM")
    
    # Arguments pour les données
    parser.add_argument("--symbol", type=str, default="BTCUSDT", 
                      help="Paire de trading (ex: BTCUSDT)")
    parser.add_argument("--timeframe", type=str, default="15m",
                      help="Intervalle de temps (ex: 15m, 1h)")
    parser.add_argument("--data_path", type=str, required=True,
                      help="Répertoire des données de marché")
    
    # Arguments pour le modèle
    parser.add_argument("--output", type=str, default=None,
                      help="Chemin où sauvegarder le modèle (défaut: data/models/<symbol>_<timeframe>.keras)")
    parser.add_argument("--lstm_units", type=str, default="128,64,32",
                      help="Unités LSTM par couche (séparées par virgules)")
    parser.add_argument("--dropout", type=float, default=0.3,
                      help="Taux de dropout")
    parser.add_argument("--learning_rate", type=float, default=0.001,
                      help="Taux d'apprentissage")
    
    # Arguments pour l'entraînement
    parser.add_argument("--epochs", type=int, default=100,
                      help="Nombre d'époques d'entraînement")
    parser.add_argument("--batch_size", type=int, default=64,
                      help="Taille des batchs d'entraînement")
    parser.add_argument("--validation_split", type=float, default=0.2,
                      help="Proportion des données pour la validation")
    
    # Options avancées
    parser.add_argument("--use_attention", type=bool, default=True,
                      help="Utiliser le mécanisme d'attention")
    parser.add_argument("--use_early_stopping", type=bool, default=True,
                      help="Utiliser l'arrêt précoce")
    parser.add_argument("--patience", type=int, default=15,
                      help="Patience pour l'arrêt précoce")
    parser.add_argument("--verbose", type=int, default=1,
                      help="Niveau de verbosité (0=silencieux, 1=progrès, 2=détaillé)")
    parser.add_argument("--evaluate_after", type=bool, default=True,
                      help="Évaluer le modèle après l'entraînement")
    
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
    # Extraire les horizons de prédiction typiques (3h, 6h, 24h pour des bougies de 15min)
    horizons = [12, 24, 96]
    if args.timeframe == '1h':
        horizons = [3, 6, 24]
    elif args.timeframe == '4h':
        horizons = [1, 3, 6]
    
    # Créer les séquences d'entrée et les cibles
    logger.info("Création des séquences pour l'entraînement...")
    X_train, y_train = feature_engineering.create_multi_horizon_data(
        train_data, 
        sequence_length=60,  # 60 périodes d'historique
        horizons=horizons,
        is_training=True
    )
    
    logger.info("Création des séquences pour la validation...")
    X_val, y_val = feature_engineering.create_multi_horizon_data(
        val_data, 
        sequence_length=60,  # 60 périodes d'historique
        horizons=horizons,
        is_training=True
    )
    
    # 4. Convertir les string d'unités LSTM en liste d'entiers
    lstm_units = [int(u) for u in args.lstm_units.split(',')]
    
    # 5. Créer et compiler le modèle
    logger.info(f"Création du modèle LSTM avec unités: {lstm_units}")
    model = LSTMModel(
        input_length=60,
        feature_dim=X_train.shape[2],
        lstm_units=lstm_units,
        dropout_rate=args.dropout,
        learning_rate=args.learning_rate,
        use_attention=args.use_attention,
        prediction_horizons=horizons
    )
    
    # 6. Préparer les callbacks
    callbacks = []
    
    # Créer le répertoire pour les checkpoints si nécessaire
    os.makedirs(MODEL_CHECKPOINTS_DIR, exist_ok=True)
    
    # Chemin pour sauvegarder le modèle
    if args.output:
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
    if args.use_early_stopping:
        callbacks.append(
            EarlyStopping(
                monitor='val_loss',
                patience=args.patience,
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
    
    # 7. Entraîner le modèle
    logger.info(f"Début de l'entraînement pour {args.epochs} époques avec batch_size={args.batch_size}...")
    history = model.model.fit(
        x=X_train,
        y=y_train,
        epochs=args.epochs,
        batch_size=args.batch_size,
        validation_data=(X_val, y_val),
        callbacks=callbacks,
        verbose=args.verbose
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
    if args.evaluate_after:
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
    
    # Calculer les métriques de direction (précision de la prédiction de direction)
    direction_accuracy = {}
    
    for i, horizon_idx in enumerate(range(0, len(predictions))):
        # Obtenir les prédictions de direction
        y_true = y_val[horizon_idx].flatten()
        y_pred = (predictions[horizon_idx].flatten() > 0.5).astype(int)
        
        # Calculer la précision
        accuracy = np.mean(y_true == y_pred)
        direction_accuracy[f"horizon_{i+1}"] = accuracy
    
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
    
    # 1. Direction prediction accuracy per horizon
    plt.figure(figsize=(10, 6))
    horizons = []
    accuracies = []
    
    for i, horizon_idx in enumerate(range(len(predictions))):
        y_true = y_val[horizon_idx].flatten()
        y_pred = (predictions[horizon_idx].flatten() > 0.5).astype(int)
        accuracy = np.mean(y_true == y_pred) * 100
        
        horizons.append(f"H{i+1}")
        accuracies.append(accuracy)
    
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
    
    # 2. Confusion matrix for the first horizon
    y_true = y_val[0].flatten()
    y_pred = (predictions[0].flatten() > 0.5).astype(int)
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
    plt.title(f'Matrice de Confusion - {symbol} {timeframe}')
    plt.ylabel('Vrai')
    plt.xlabel('Prédit')
    plt.savefig(os.path.join(viz_dir, f"{symbol}_{timeframe}_confusion_matrix.png"))
    plt.close()
    
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