#!/usr/bin/env python
# train_model.py
"""
Script d'entraînement du modèle LSTM pour la prédiction des mouvements de marché
"""
import os
import argparse
import pandas as pd
from sklearn.utils import class_weight
import numpy as np
from datetime import datetime, timedelta
import json
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau, LambdaCallback
import tensorflow.keras.backend as K

from ai.models.lstm_model import LSTMModel, EnhancedLSTMModel
from ai.models.feature_engineering import FeatureEngineering
from ai.models.model_trainer import ModelTrainer
from ai.models.model_validator import ModelValidator
from config.config import DATA_DIR, MODEL_CHECKPOINTS_DIR
from utils.logger import setup_logger
# Remove the problematic import
# from download_data import load_historical_data
from sklearn.utils import class_weight

logger = setup_logger("train_model")

def load_data(symbol: str, timeframe: str, start_date: str, end_date: str) -> pd.DataFrame:
    """
    Charge les données OHLCV depuis le disque
    
    Args:
        symbol: Paire de trading
        timeframe: Intervalle de temps
        start_date: Date de début (YYYY-MM-DD)
        end_date: Date de fin (YYYY-MM-DD)
        
    Returns:
        DataFrame avec les données OHLCV
    """
    # Construire le chemin du fichier
    data_path = os.path.join(DATA_DIR, "market_data", f"{symbol}_{timeframe}_{start_date}_{end_date}.csv")
    
    # Vérifier si le fichier existe
    if not os.path.exists(data_path):
        logger.error(f"Fichier non trouvé: {data_path}")
        return pd.DataFrame()
    
    # Charger les données
    try:
        data = pd.read_csv(data_path)
        
        # Convertir la colonne timestamp en datetime
        if "timestamp" in data.columns:
            data["timestamp"] = pd.to_datetime(data["timestamp"])
            data.set_index("timestamp", inplace=True)
        
        logger.info(f"Données chargées: {len(data)} lignes")
        return data
    except Exception as e:
        logger.error(f"Erreur lors du chargement des données: {str(e)}")
        return pd.DataFrame()

# Add this function to replace the missing load_historical_data function
def load_historical_data(symbol: str, timeframe: str) -> pd.DataFrame:
    """
    Charge les données historiques disponibles pour un symbole et un timeframe
    
    Args:
        symbol: Paire de trading (ex: 'BTCUSDT')
        timeframe: Intervalle de temps (ex: '15m')
        
    Returns:
        DataFrame avec les données OHLCV ou DataFrame vide si erreur
    """
    # Chercher les fichiers de données correspondants dans le répertoire des données
    market_data_dir = os.path.join(DATA_DIR, "market_data")
    if not os.path.exists(market_data_dir):
        logger.error(f"Répertoire de données non trouvé: {market_data_dir}")
        return pd.DataFrame()
    
    # Recherche les fichiers qui correspondent au pattern du symbole et timeframe
    matching_files = [f for f in os.listdir(market_data_dir) 
                     if f.startswith(f"{symbol}_{timeframe}") and f.endswith('.csv')]
    
    if not matching_files:
        logger.error(f"Aucun fichier de données trouvé pour {symbol} ({timeframe})")
        return pd.DataFrame()
    
    # Utiliser le fichier le plus récent
    latest_file = sorted(matching_files)[-1]
    file_path = os.path.join(market_data_dir, latest_file)
    
    logger.info(f"Chargement du fichier de données: {file_path}")
    return load_data(symbol, timeframe, "", "")  # Les dates sont ignorées car on utilise le chemin direct

def download_data_if_needed(symbol: str, timeframe: str, start_date: str, end_date: str) -> bool:
    """
    Télécharge les données si elles n'existent pas sur le disque
    
    Args:
        symbol: Paire de trading
        timeframe: Intervalle de temps
        start_date: Date de début (YYYY-MM-DD)
        end_date: Date de fin (YYYY-MM-DD)
        
    Returns:
        True si les données sont disponibles, False sinon
    """
    # Construire le chemin du fichier
    data_path = os.path.join(DATA_DIR, "market_data", f"{symbol}_{timeframe}_{start_date}_{end_date}.csv")
    
    # Vérifier si le fichier existe
    if os.path.exists(data_path):
        logger.info(f"Données déjà disponibles: {data_path}")
        return True
    
    # Télécharger les données
    logger.info(f"Téléchargement des données pour {symbol} ({timeframe}) du {start_date} au {end_date}")
    
    try:
        from download_data import download_binance_data
        
        # Télécharger les données
        df = download_binance_data(symbol, timeframe, start_date, end_date, data_path)
        
        if df is not None and not df.empty:
            logger.info(f"Données téléchargées avec succès: {len(df)} lignes")
            return True
        else:
            logger.error("Échec du téléchargement des données")
            return False
    except Exception as e:
        logger.error(f"Erreur lors du téléchargement des données: {str(e)}")
        return False

def train_lstm_model(args):
    """
    Entraîne le modèle LSTM avec les paramètres spécifiés
    
    Args:
        args: Arguments de ligne de commande
    """
    # Télécharger les données si nécessaire
    if not download_data_if_needed(args.symbol, args.timeframe, args.start_date, args.end_date):
        logger.error("Impossible de continuer sans données")
        return
    
    # Charger les données
    data = load_data(args.symbol, args.timeframe, args.start_date, args.end_date)
    
    if data.empty:
        logger.error("Données vides, impossible de continuer")
        return
    
    # Configurer les paramètres du modèle
    model_params = {
        "input_length": args.sequence_length,
        "feature_dim": args.feature_dim,
        "lstm_units": [args.lstm_units, args.lstm_units // 2, args.lstm_units // 4],
        "dropout_rate": args.dropout,
        "learning_rate": args.learning_rate,
        "l1_reg": args.l1_reg,
        "l2_reg": args.l2_reg,
        "use_attention": not args.no_attention,
        "use_residual": not args.no_residual,
        "prediction_horizons": [args.short_horizon, args.mid_horizon, args.long_horizon]
    }
    
    logger.info(f"Configuration du modèle: {json.dumps(model_params, indent=2)}")
    
    # Créer le ModelTrainer
    trainer = ModelTrainer(model_params)
    
    # Préparer les données
    logger.info("Préparation des données...")
    featured_data, normalized_data = trainer.prepare_data(data)
    
    # Calculer les poids des classes
    y_direction = ...  # Assurez-vous que y_direction est défini dans votre préparation de données
    class_weights = class_weight.compute_class_weight(
        class_weight='balanced',
        classes=np.unique(y_direction),
        y=y_direction
    )
    class_weight_dict = {i: weight for i, weight in enumerate(class_weights)}
    
    # Ajouter des callbacks pour une meilleure régularisation
    callbacks = [
        EarlyStopping(
            monitor='val_loss',
            patience=15,
            restore_best_weights=True
        ),
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=8,
            min_lr=1e-6
        ),
        # Augmenter le dropout progressivement si nécessaire
        LambdaCallback(
            on_epoch_end=lambda epoch, logs: K.set_value(
                trainer.model.get_layer('dropout_layer').rate, 
                min(0.5, 0.3 + epoch * 0.01)
            ) if epoch > 10 else None
        )
    ]
    
    # Diviser les données en ensembles d'entraînement, validation et test
    if args.cv:
        # Entraînement avec validation croisée
        logger.info("Entraînement avec validation croisée temporelle...")
        cv_results = trainer.train_with_cv(
            normalized_data,
            n_splits=args.cv_splits,
            epochs=args.epochs,
            batch_size=args.batch_size,
            initial_train_ratio=args.train_ratio,
            patience=args.patience,
            callbacks=callbacks
        )
        
        logger.info(f"Entraînement terminé, perte moyenne de validation: {cv_results['avg_val_loss']:.4f}")
    else:
        # Entraînement simple avec division train/val/test
        logger.info("Entraînement standard avec division temporelle...")
        train_data, val_data, test_data = trainer.temporal_train_test_split(
            normalized_data,
            train_ratio=args.train_ratio,
            val_ratio=args.val_ratio
        )
        
        # Créer et entraîner le modèle
        train_results = trainer.train_final_model(
            normalized_data,
            epochs=args.epochs,
            batch_size=args.batch_size,
            test_ratio=1.0 - args.train_ratio - args.val_ratio,
            class_weight=class_weight_dict,
            callbacks=callbacks
        )
        
        logger.info(f"Entraînement terminé, perte sur le test: {train_results['test_loss']:.4f}")
        
        # Afficher les précisions de direction par horizon
        for i, horizon in enumerate(model_params["prediction_horizons"]):
            accuracy = train_results["direction_accuracies"][i]
            logger.info(f"Précision de direction pour horizon {horizon}: {accuracy:.2f}")
    
    # Valider le modèle final sur des données récentes
    if args.validate:
        logger.info("Validation du modèle sur des données récentes...")
        
        # Charger des données récentes pour la validation
        end_date = datetime.now().strftime("%Y-%m-%d")
        start_date = (datetime.now() - timedelta(days=30)).strftime("%Y-%m-%d")
        
        if download_data_if_needed(args.symbol, args.timeframe, start_date, end_date):
            validation_data = load_data(args.symbol, args.timeframe, start_date, end_date)
            
            if not validation_data.empty:
                # Créer le validateur
                validator = ModelValidator(trainer.model, trainer.feature_engineering)
                
                # Évaluer sur les données récentes
                validation_results = validator.evaluate_on_test_set(validation_data)
                
                logger.info(f"Validation terminée, perte: {validation_results['loss']:.4f}")
                
                # Afficher les métriques par horizon
                for horizon_key, metrics in validation_results["horizons"].items():
                    direction_acc = metrics["direction"]["accuracy"]
                    direction_f1 = metrics["direction"]["f1_score"]
                    
                    logger.info(f"{horizon_key}: Accuracy={direction_acc:.2f}, F1={direction_f1:.2f}")
    
    logger.info("Processus d'entraînement terminé")
    logger.info(f"Modèle final sauvegardé: {os.path.join(DATA_DIR, 'models', 'production', 'lstm_final.h5')}")

def evaluate_model(args):
    """
    Évalue un modèle LSTM existant sur de nouvelles données
    
    Args:
        args: Arguments de ligne de commande
    """
    # Télécharger les données de test si nécessaire
    if not download_data_if_needed(args.symbol, args.timeframe, args.start_date, args.end_date):
        logger.error("Impossible de continuer sans données")
        return
    
    # Charger les données
    data = load_data(args.symbol, args.timeframe, args.start_date, args.end_date)
    
    if data.empty:
        logger.error("Données vides, impossible de continuer")
        return
    
    # Charger le modèle existant
    model_path = args.model_path or os.path.join(DATA_DIR, "models", "production", "lstm_final.h5")
    
    if not os.path.exists(model_path):
        logger.error(f"Modèle non trouvé: {model_path}")
        return
    
    # Créer le validateur
    validator = ModelValidator()
    validator.load_model(model_path)
    
    # Évaluer le modèle
    evaluation = validator.evaluate_on_test_set(data)
    
    logger.info(f"Évaluation terminée, perte globale: {evaluation['loss']:.4f}")
    
    # Afficher les métriques par horizon
    for horizon_key, metrics in evaluation["horizons"].items():
        direction_metrics = metrics["direction"]
        volatility_metrics = metrics["volatility"]
        
        logger.info(f"\nHorizon: {horizon_key}")
        logger.info(f"Direction: Accuracy={direction_metrics['accuracy']:.2f}, "
                   f"Precision={direction_metrics['precision']:.2f}, "
                   f"Recall={direction_metrics['recall']:.2f}, "
                   f"F1={direction_metrics['f1_score']:.2f}")
        logger.info(f"Volatilité: MAE={volatility_metrics['mae']:.4f}, "
                   f"RMSE={volatility_metrics['rmse']:.4f}")
    
    # Comparaison avec la stratégie de base
    if args.compare:
        logger.info("\nComparaison avec la stratégie de base...")
        
        comparison = validator.compare_with_baseline(
            data,
            initial_capital=args.capital
        )
        
        # Afficher les résultats
        baseline = comparison["baseline"]
        lstm = comparison["lstm"]
        diff = comparison["comparison"]
        
        logger.info("\n=== Résultats de la comparaison ===")
        logger.info(f"Stratégie de base: {baseline['return_pct']:.2f}% (Drawdown: {baseline['max_drawdown_pct']:.2f}%, Sharpe: {baseline['sharpe_ratio']:.2f})")
        logger.info(f"Modèle LSTM: {lstm['return_pct']:.2f}% (Drawdown: {lstm['max_drawdown_pct']:.2f}%, Sharpe: {lstm['sharpe_ratio']:.2f})")
        logger.info(f"Différence: {diff['return_difference']:.2f}%, Amélioration drawdown: {diff['drawdown_improvement']:.2f}%, Amélioration Sharpe: {diff['sharpe_improvement']:.2f}")
        
        # Sauvegarder les résultats
        comparison_file = os.path.join(DATA_DIR, "models", "evaluation", f"comparison_{args.symbol}_{datetime.now().strftime('%Y%m%d')}.json")
        os.makedirs(os.path.dirname(comparison_file), exist_ok=True)
        
        with open(comparison_file, 'w') as f:
            json.dump(comparison, f, indent=2, default=str)
        
        logger.info(f"Résultats de comparaison sauvegardés: {comparison_file}")

def main():
    """Point d'entrée principal du script"""
    parser = argparse.ArgumentParser(description="Entraînement et évaluation du modèle LSTM")
    
    subparsers = parser.add_subparsers(dest="command", help="Commande à exécuter")
    
    # Parser pour l'entraînement
    train_parser = subparsers.add_parser("train", help="Entraîner un nouveau modèle LSTM")
    
    # Arguments pour les données
    train_parser.add_argument("--symbol", type=str, default="BTCUSDT", help="Paire de trading")
    train_parser.add_argument("--timeframe", type=str, default="15m", help="Intervalle de temps")
    train_parser.add_argument("--start-date", type=str, required=True, help="Date de début (YYYY-MM-DD)")
    train_parser.add_argument("--end-date", type=str, required=True, help="Date de fin (YYYY-MM-DD)")
    
    # Arguments pour le modèle
    train_parser.add_argument("--sequence-length", type=int, default=60, help="Longueur des séquences d'entrée")
    train_parser.add_argument("--feature-dim", type=int, default=30, help="Dimension des caractéristiques")
    train_parser.add_argument("--lstm-units", type=int, default=128, help="Nombre d'unités LSTM")
    train_parser.add_argument("--dropout", type=float, default=0.3, help="Taux de dropout")
    train_parser.add_argument("--learning-rate", type=float, default=0.001, help="Taux d'apprentissage")
    train_parser.add_argument("--l1-reg", type=float, default=0.0001, help="Régularisation L1")
    train_parser.add_argument("--l2-reg", type=float, default=0.0001, help="Régularisation L2")
    train_parser.add_argument("--no-attention", action="store_true", help="Désactiver le mécanisme d'attention")
    train_parser.add_argument("--no-residual", action="store_true", help="Désactiver les connexions résiduelles")
    
    # Horizons de prédiction
    train_parser.add_argument("--short-horizon", type=int, default=12, help="Horizon court terme (nb de périodes)")
    train_parser.add_argument("--mid-horizon", type=int, default=24, help="Horizon moyen terme (nb de périodes)")
    train_parser.add_argument("--long-horizon", type=int, default=96, help="Horizon long terme (nb de périodes)")
    
    # Arguments pour l'entraînement
    train_parser.add_argument("--epochs", type=int, default=100, help="Nombre d'époques")
    train_parser.add_argument("--batch-size", type=int, default=32, help="Taille du batch")
    train_parser.add_argument("--patience", type=int, default=20, help="Patience pour l'early stopping")
    train_parser.add_argument("--train-ratio", type=float, default=0.7, help="Ratio des données d'entraînement")
    train_parser.add_argument("--val-ratio", type=float, default=0.15, help="Ratio des données de validation")
    
    # Validation croisée
    train_parser.add_argument("--cv", action="store_true", help="Utiliser la validation croisée temporelle")
    train_parser.add_argument("--cv-splits", type=int, default=5, help="Nombre de plis pour la validation croisée")
    
    # Validation finale
    train_parser.add_argument("--validate", action="store_true", help="Valider sur des données récentes après l'entraînement")
    
    # Parser pour l'évaluation
    eval_parser = subparsers.add_parser("evaluate", help="Évaluer un modèle LSTM existant")
    
    # Arguments pour les données
    eval_parser.add_argument("--symbol", type=str, default="BTCUSDT", help="Paire de trading")
    eval_parser.add_argument("--timeframe", type=str, default="15m", help="Intervalle de temps")
    eval_parser.add_argument("--start-date", type=str, required=True, help="Date de début (YYYY-MM-DD)")
    eval_parser.add_argument("--end-date", type=str, required=True, help="Date de fin (YYYY-MM-DD)")
    
    # Arguments pour le modèle
    eval_parser.add_argument("--model-path", type=str, help="Chemin vers le modèle à évaluer")
    
    # Comparaison avec stratégie de base
    eval_parser.add_argument("--compare", action="store_true", help="Comparer avec la stratégie de base")
    eval_parser.add_argument("--capital", type=float, default=200, help="Capital initial pour la comparaison")
    
    args = parser.parse_args()
    
    if args.command == "train":
        train_lstm_model(args)
    elif args.command == "evaluate":
        evaluate_model(args)
    else:
        parser.print_help()

if __name__ == "__main__":
    main()