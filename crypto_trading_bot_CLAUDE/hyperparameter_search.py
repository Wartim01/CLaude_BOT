#!/usr/bin/env python
# hyperparameter_search.py
"""
Script d'optimisation des hyperparamètres pour le modèle LSTM
Utilise Optuna pour une recherche bayésienne efficace
"""
import os
import argparse
import pandas as pd
import numpy as np
import optuna
import tensorflow as tf
from datetime import datetime, timedelta
import json
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score
import re

from ai.models.lstm_model import LSTMModel
from ai.models.feature_engineering import FeatureEngineering
from ai.models.model_trainer import ModelTrainer
from ai.models.model_validator import ModelValidator
from config.config import DATA_DIR
from utils.logger import setup_logger
from train_model import load_data
from config.model_params import LSTM_DEFAULT_PARAMS

# Configuration du logger
logger = setup_logger("hyperparameter_search")

def update_model_params_file(best_params, timeframe, f1_score):
    """
    Updates the parameters in config.model_params.py with the best parameters
    found during hyperparameter optimization
    
    Args:
        best_params: Dictionary containing the best hyperparameters
        timeframe: Current timeframe (e.g. 15m, 1h)
        f1_score: F1 score achieved by the best parameters
    """
    # Path to the model_params.py file
    params_file = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 
                             'config', 'model_params.py')
    
    if not os.path.exists(params_file):
        logger.error(f"Configuration file not found: {params_file}")
        return False
    
    try:
        # Read the current file content
        with open(params_file, 'r') as file:
            content = file.read()
        
        # Update both LSTM_DEFAULT_PARAMS and LSTM_OPTIMIZED_PARAMS
        
        # 1. First update the default params as before
        new_params = {
            "lstm_units": [best_params.get("lstm_units_first", 128)] + 
                         [best_params.get("lstm_units_first", 128) // 2] * (best_params.get("lstm_layers", 2) - 1),
            "dropout_rate": best_params.get("dropout_rate", 0.3),
            "learning_rate": best_params.get("learning_rate", 0.001),
            "batch_size": best_params.get("batch_size", 64),
            "sequence_length": best_params.get("sequence_length", 60),
            "l1_regularization": best_params.get("l1_reg", 0.0001),
            "l2_regularization": best_params.get("l2_reg", 0.0001)
        }
        
        # 2. Now update the optimized params for this specific timeframe
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        # Convert the lstm_units to a proper string representation for the config file
        lstm_units_str = str([best_params.get("lstm_units_first", 128)] + 
                            [best_params.get("lstm_units_first", 128) // 2] * (best_params.get("lstm_layers", 2) - 1))
        
        # Create pattern to find and replace the relevant timeframe section inside LSTM_OPTIMIZED_PARAMS
        pattern = rf'"{timeframe}":\s*\{{[^}}]*\}}'
        
        # Create the replacement text
        replacement = f'"{timeframe}": {{\n'
        replacement += f'        "lstm_units": {lstm_units_str},\n'
        replacement += f'        "dropout_rate": {best_params.get("dropout_rate", 0.3)},\n'
        replacement += f'        "learning_rate": {best_params.get("learning_rate", 0.001)},\n'
        replacement += f'        "batch_size": {best_params.get("batch_size", 64)},\n'
        replacement += f'        "sequence_length": {best_params.get("sequence_length", 60)},\n'
        replacement += f'        "l1_regularization": {best_params.get("l1_reg", 0.0001)},\n'
        replacement += f'        "l2_regularization": {best_params.get("l2_reg", 0.0001)},\n'
        replacement += f'        "use_attention": True,\n'
        replacement += f'        "use_residual": True,\n'
        replacement += f'        "last_optimized": "{timestamp}",\n'
        replacement += f'        "f1_score": {f1_score}\n'
        replacement += f'    }}'
        
        # Update the LSTM_OPTIMIZED_PARAMS section
        updated_content = re.sub(pattern, replacement, content)
        
        # Also update the "Last optimization" comment
        updated_content = re.sub(r'# Last optimization: .*', f'# Last optimization: {timestamp} (timeframe: {timeframe}, F1: {f1_score:.4f})', updated_content)
        
        # Also update the default params as before
        default_pattern = r'LSTM_DEFAULT_PARAMS\s*=\s*\{[^}]*\}'
        new_params_str = "LSTM_DEFAULT_PARAMS = {\n"
        for key, value in new_params.items():
            if isinstance(value, str):
                new_params_str += f'    "{key}": "{value}",\n'
            else:
                new_params_str += f'    "{key}": {value},\n'
        new_params_str += "    # Other parameters preserved\n"
        new_params_str += '    "use_attention": True,\n'
        new_params_str += '    "use_residual": True,\n'
        new_params_str += '    "epochs": 100,\n'
        new_params_str += '    "early_stopping_patience": 15,\n'
        new_params_str += '    "reduce_lr_patience": 7\n'
        new_params_str += "}"
        
        updated_content = re.sub(default_pattern, new_params_str, updated_content, flags=re.DOTALL)
        
        # Write the updated content back to the file
        with open(params_file, 'w') as file:
            file.write(updated_content)
        
        logger.info(f"Parameters in {params_file} updated successfully with F1 score: {f1_score:.4f}")
        return True
        
    except Exception as e:
        logger.error(f"Error updating model parameters file: {str(e)}")
        return False

class BestModelCallback:
    """
    Callback for Optuna to update model_params.py when a new best trial is found
    """
    def __init__(self, timeframe):
        self.best_value = -float('inf')
        self.best_params = None
        self.timeframe = timeframe
    
    def __call__(self, study, trial):
        # Check if we found a new best trial
        if study.best_value > self.best_value:
            self.best_value = study.best_value
            self.best_params = study.best_params
            
            logger.info(f"New best trial found! Trial {trial.number}: F1={study.best_value:.4f}")
            
            # Update the model_params.py file with the new best parameters
            update_model_params_file(self.best_params, self.timeframe, self.best_value)

class LSTMHyperparameterOptimizer:
    """
    Classe pour l'optimisation des hyperparamètres du modèle LSTM
    """
    def __init__(self, train_data, val_data, symbol="BTCUSDT", timeframe="15m"):
        """
        Initialise l'optimiseur d'hyperparamètres
        
        Args:
            train_data: Données d'entraînement
            val_data: Données de validation
            symbol: Symbole de la paire de trading
            timeframe: Intervalle de temps
        """
        self.train_data = train_data
        self.val_data = val_data
        self.symbol = symbol
        self.timeframe = timeframe
        
        # Répertoire pour sauvegarder les résultats
        self.output_dir = os.path.join(DATA_DIR, "models", "optimization")
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Feature engineering partagé entre toutes les tentatives
        self.feature_engineering = FeatureEngineering(save_scalers=True)
        
        # Préparer les données une seule fois
        logger.info("Préparation des données pour l'optimisation...")
        self.featured_train, self.normalized_train = self.feature_engineering.create_features(train_data), self.feature_engineering.scale_features(
            self.feature_engineering.create_features(train_data),
            is_training=True,
            method='standard',
            feature_group='lstm_optim'
        )
        
        self.featured_val, self.normalized_val = self.feature_engineering.create_features(val_data), self.feature_engineering.scale_features(
            self.feature_engineering.create_features(val_data),
            is_training=False,
            method='standard',
            feature_group='lstm_optim'
        )
        
        # Historique des essais
        self.trials_history = []
    
    def objective(self, trial):
        """
        Fonction objectif pour Optuna avec modèle à sortie unique
        """
        try:
            # Configuration des paramètres
            lstm_units_first = trial.suggest_int("lstm_units_first", 64, 256)
            lstm_layers = trial.suggest_int("lstm_layers", 1, 3)
            dropout_rate = trial.suggest_float("dropout_rate", 0.1, 0.5)
            learning_rate = trial.suggest_float("learning_rate", 1e-5, 1e-2, log=True)
            
            # Construction des unités LSTM en fonction du nombre de couches
            lstm_units = [lstm_units_first]
            for i in range(1, lstm_layers):
                lstm_units.append(lstm_units[-1] // 2)
            
            # Process a small batch of data to determine the actual feature dimension
            X_sample, _ = self.feature_engineering.create_multi_horizon_data(
                self.normalized_train.iloc[:100],  # Use a small sample
                sequence_length=60,  # Default value
                horizons=[4],  # For 1h prediction with 15m data
                is_training=True
            )
            
            # Get the actual feature dimension from the processed data
            actual_feature_dim = X_sample.shape[2]
            logger.info(f"Detected actual feature dimension: {actual_feature_dim}")
            
            # Paramètres du modèle with dynamic feature dimension
            model_params = {
                "input_length": trial.suggest_categorical("sequence_length", [30, 60, 90]),
                "feature_dim": actual_feature_dim,  # Use actual dimension
                "lstm_units": lstm_units,
                "dropout_rate": dropout_rate,
                "learning_rate": learning_rate,
                "l1_reg": trial.suggest_float("l1_reg", 1e-6, 1e-3, log=True),
                "l2_reg": trial.suggest_float("l2_reg", 1e-6, 1e-3, log=True),
                "use_attention": False,  # Simplifier pour l'optimisation
                "use_residual": False,   # Simplifier pour l'optimisation
                "prediction_horizons": [4]  # Single horizon for 1h with 15m data
            }
            
            logger.info(f"Essai {trial.number}: {json.dumps(model_params, indent=2)}")
            
            # Créer le trainer avec les paramètres
            trainer = ModelTrainer(model_params)
            
            # Créer un modèle à sortie unique pour l'optimisation
            single_model = trainer.model.build_single_output_model(horizon_idx=0)  # Utiliser l'horizon court terme
            
            # Préparer les données pour ce modèle
            X_train, y_lists = trainer.feature_engineering.create_multi_horizon_data(
                self.normalized_train,
                sequence_length=model_params["input_length"],
                horizons=model_params["prediction_horizons"],
                is_training=True
            )
            
            X_val, y_val_lists = trainer.feature_engineering.create_multi_horizon_data(
                self.normalized_val,
                sequence_length=model_params["input_length"],
                horizons=model_params["prediction_horizons"],
                is_training=True
            )
            
            # Extraire uniquement les directions pour l'horizon court terme
            y_train = y_lists[0]  # Direction pour le premier horizon
            y_val = y_val_lists[0]  # Direction pour le premier horizon
            
            # Calculer les poids des classes pour l'équilibrage
            class_counts = np.bincount(y_train.flatten())
            class_weights = {
                0: len(y_train) / (2 * class_counts[0]),
                1: len(y_train) / (2 * class_counts[1])
            }
            
            # Définir les callbacks
            callbacks = [
                tf.keras.callbacks.EarlyStopping(
                    monitor='val_accuracy',
                    patience=5,
                    restore_best_weights=True
                ),
                tf.keras.callbacks.ReduceLROnPlateau(
                    monitor='val_loss',
                    factor=0.5,
                    patience=3,
                    min_lr=1e-6
                )
            ]
            
            # Entraîner le modèle
            batch_size = trial.suggest_categorical("batch_size", [16, 32, 64, 128])
            history = single_model.fit(
                x=X_train,
                y=y_train,
                epochs=30,  # Réduit pour l'optimisation
                batch_size=batch_size,
                validation_data=(X_val, y_val),
                class_weight=class_weights,
                callbacks=callbacks,
                verbose=0
            )
            
            # Évaluer le modèle
            val_loss, val_accuracy = single_model.evaluate(X_val, y_val, verbose=0)
            
            # Prédictions pour calculer le F1 score
            y_pred = (single_model.predict(X_val) > 0.5).astype(int).flatten()
            f1 = f1_score(y_val, y_pred)
            
            # Enregistrer les résultats
            trial_result = {
                "trial_number": trial.number,
                "params": model_params,
                "batch_size": batch_size,
                "val_loss": float(val_loss),
                "val_accuracy": float(val_accuracy),
                "f1_score": float(f1),
                "class_weights": {str(k): float(v) for k, v in class_weights.items()},
                "timestamp": datetime.now().isoformat()
            }
            
            self.trials_history.append(trial_result)
            
            # Sauvegarder périodiquement
            if len(self.trials_history) % 5 == 0:
                self._save_trials_history()
            
            logger.info(f"Essai {trial.number}: F1={f1:.4f}, Accuracy={val_accuracy:.4f}")
            
            return f1  # Optimiser le F1 score
            
        except Exception as e:
            logger.error(f"Erreur lors de l'entraînement: {str(e)}")
            return 0.0  # Valeur par défaut en cas d'erreur
    
    def _save_trials_history(self):
        """Sauvegarde l'historique des essais"""
        history_path = os.path.join(self.output_dir, f"trials_history_{datetime.now().strftime('%Y%m%d')}.json")
        
        try:
            with open(history_path, 'w') as f:
                json.dump(self.trials_history, f, indent=2, default=str)
            logger.info(f"Historique des essais sauvegardé: {history_path}")
        except Exception as e:
            logger.error(f"Erreur lors de la sauvegarde de l'historique: {str(e)}")
    
    def run_optimization(self, n_trials=50, timeout=None, callbacks=None):
        """
        Lance l'optimisation des hyperparamètres
        
        Args:
            n_trials: Nombre d'essais
            timeout: Temps maximum en secondes
            callbacks: Liste de callbacks Optuna
            
        Returns:
            Les meilleurs hyperparamètres trouvés
        """
        logger.info(f"Démarrage de l'optimisation avec {n_trials} essais...")
        
        # Créer l'étude Optuna
        study = optuna.create_study(
            direction="maximize",
            sampler=optuna.samplers.TPESampler(seed=42),
            pruner=optuna.pruners.MedianPruner()
        )
        
        # Préparer les callbacks
        optuna_callbacks = []
        if callbacks:
            optuna_callbacks.extend(callbacks)
        
        # Lancer l'optimisation
        study.optimize(self.objective, n_trials=n_trials, timeout=timeout, callbacks=optuna_callbacks)
        
        # Obtenir les meilleurs hyperparamètres
        best_params = study.best_params
        best_value = study.best_value
        
        logger.info(f"Optimisation terminée!")
        logger.info(f"Meilleur F1 score: {best_value:.4f}")
        logger.info(f"Meilleurs hyperparamètres: {json.dumps(best_params, indent=2)}")
        
        # Sauvegarder les meilleurs hyperparamètres
        best_params_path = os.path.join(self.output_dir, f"best_params_{datetime.now().strftime('%Y%m%d')}.json")
        
        try:
            with open(best_params_path, 'w') as f:
                json.dump({
                    "best_params": best_params,
                    "best_value": best_value,
                    "n_trials": n_trials,
                    "timestamp": datetime.now().isoformat()
                }, f, indent=2)
            logger.info(f"Meilleurs hyperparamètres sauvegardés: {best_params_path}")
            
            # Update the model_params.py file with the best parameters
            update_model_params_file(best_params, self.timeframe, best_value)
            
        except Exception as e:
            logger.error(f"Erreur lors de la sauvegarde des meilleurs hyperparamètres: {str(e)}")
        
        # Visualiser l'importance des hyperparamètres
        self._visualize_importance(study)
        
        return best_params
    
    def _visualize_importance(self, study):
        """
        Visualise l'importance des hyperparamètres
        
        Args:
            study: Étude Optuna
        """
        try:
            # Créer la figure
            plt.figure(figsize=(10, 6))
            
            # Calculer l'importance des paramètres
            importances = optuna.importance.get_param_importances(study)
            
            # Trier les paramètres par importance
            params = list(importances.keys())
            scores = list(importances.values())
            indices = np.argsort(scores)
            
            # Créer le graphique à barres horizontales
            plt.barh(range(len(indices)), [scores[i] for i in indices], color='skyblue')
            plt.yticks(range(len(indices)), [params[i] for i in indices])
            plt.xlabel('Importance relative')
            plt.title('Importance des hyperparamètres')
            plt.tight_layout()
            
            # Sauvegarder la figure
            figure_path = os.path.join(self.output_dir, f"param_importance_{datetime.now().strftime('%Y%m%d')}.png")
            plt.savefig(figure_path)
            plt.close()
            
            logger.info(f"Visualisation de l'importance des hyperparamètres sauvegardée: {figure_path}")
        except Exception as e:
            logger.error(f"Erreur lors de la visualisation de l'importance des hyperparamètres: {str(e)}")

def download_data_if_needed(symbol, timeframe, start_date, end_date):
    """
    Check if data file exists and return True if it does
    
    Args:
        symbol: Trading pair symbol
        timeframe: Time interval
        start_date: Start date for data
        end_date: End date for data
        
    Returns:
        True if data is available, False otherwise
    """
    from config.config import MARKET_DATA_DIR
    import os
    
    # Expected file path
    file_path = os.path.join(MARKET_DATA_DIR, f"{symbol}_{timeframe}.csv")
    
    # Check if file exists
    if os.path.exists(file_path):
        logger.info(f"Data file found: {file_path}")
        return True
    
    # Alternatively, look for files matching symbol and timeframe
    matching_files = [
        f for f in os.listdir(MARKET_DATA_DIR)
        if f.endswith('.csv') and symbol in f and timeframe in f
    ]
    
    if matching_files:
        logger.info(f"Found matching data file: {matching_files[0]}")
        return True
        
    logger.error(f"No data file found for {symbol}_{timeframe}. Please download the data first.")
    return False

def main():
    """Point d'entrée principal"""
    parser = argparse.ArgumentParser(description="Optimisation des hyperparamètres du modèle LSTM")
    
    # Arguments pour les données
    parser.add_argument("--symbol", type=str, default="BTCUSDT", help="Paire de trading")
    parser.add_argument("--timeframe", type=str, default="15m", help="Intervalle de temps")
    
    # Support flexible argument naming (both hyphen and underscore)
    parser.add_argument("--start-date", "--start_date", "--start", type=str, required=True, 
                      help="Date de début (YYYY-MM-DD)")
    parser.add_argument("--end-date", "--end_date", "--end", type=str, required=True, 
                      help="Date de fin (YYYY-MM-DD)")
    parser.add_argument("--data_path", "--data-path", type=str, default="data/market_data/", 
                      help="Répertoire des données de marché")
    
    # Arguments pour l'optimisation
    parser.add_argument("--n-trials", "--n_trials", type=int, default=50, 
                      help="Nombre d'essais")
    parser.add_argument("--timeout", type=int, default=None, 
                      help="Temps maximum en secondes")
    parser.add_argument("--train-ratio", "--train_ratio", type=float, default=0.7, 
                      help="Ratio des données d'entraînement")
    parser.add_argument("--val-ratio", "--val_ratio", type=float, default=0.15, 
                      help="Ratio des données de validation")
    
    args = parser.parse_args()
    
    # Check for data availability
    if not download_data_if_needed(args.symbol, args.timeframe, args.start_date, args.end_date):
        logger.error("Impossible de continuer sans données")
        return
    
    # Charger les données avec les bons paramètres
    # load_data prend data_path, symbol et timeframe comme paramètres
    data_path = os.path.join(DATA_DIR, "market_data") if args.data_path == "data/market_data/" else args.data_path
    data = load_data(data_path, args.symbol, args.timeframe)
    
    # Filtrer les données selon les dates si spécifiées
    if args.start_date and args.end_date:
        start_date = pd.to_datetime(args.start_date)
        end_date = pd.to_datetime(args.end_date)
        logger.info(f"Filtrage des données entre {start_date} et {end_date}")
        data = data[(data.index >= start_date) & (data.index <= end_date)]
    
    if data.empty:
        logger.error("Données vides, impossible de continuer")
        return
    
    # Diviser les données en ensembles d'entraînement et de validation
    train_size = int(len(data) * args.train_ratio)
    val_size = int(len(data) * args.val_ratio)
    
    train_data = data.iloc[:train_size]
    val_data = data.iloc[train_size:train_size+val_size]
    
    logger.info(f"Données divisées: {len(train_data)} lignes pour l'entraînement, {len(val_data)} lignes pour la validation")
    
    # Create the best model callback with proper timeframe
    best_model_callback = BestModelCallback(args.timeframe)
    
    # Créer l'optimiseur
    optimizer = LSTMHyperparameterOptimizer(
        train_data=train_data,
        val_data=val_data,
        symbol=args.symbol,
        timeframe=args.timeframe
    )
    
    # Lancer l'optimisation with the callback explicitly passed
    best_params = optimizer.run_optimization(
        n_trials=args.n_trials,
        timeout=args.timeout,
        callbacks=[best_model_callback]  # Make sure callback is passed here
    )
    
    # Afficher un résumé
    logger.info("\n=== Résumé de l'optimisation ===")
    logger.info(f"Meilleurs hyperparamètres: {json.dumps(best_params, indent=2)}")
    logger.info("Utilisez ces paramètres pour entraîner votre modèle final:")
    
    # Construire la commande pour l'entraînement
    lstm_units = best_params.get("lstm_units_first", 128)
    
    cmd = f"python train_model.py train --symbol {args.symbol} --timeframe {args.timeframe} "
    cmd += f"--start-date {args.start_date} --end-date {args.end_date} "
    cmd += f"--sequence-length {best_params.get('sequence_length', 60)} "
    cmd += f"--lstm-units {lstm_units} "
    cmd += f"--dropout {best_params.get('dropout_rate', 0.3)} "
    cmd += f"--learning-rate {best_params.get('learning_rate', 0.001)} "
    cmd += f"--l1-reg {best_params.get('l1_reg', 0.0001)} "
    cmd += f"--l2-reg {best_params.get('l2_reg', 0.0001)} "
    cmd += f"--batch-size {best_params.get('batch_size', 32)} "
    
    if not best_params.get("use_attention", True):
        cmd += "--no-attention "
    
    if not best_params.get("use_residual", True):
        cmd += "--no-residual "
    
    logger.info(f"Commande suggérée: {cmd}")

if __name__ == "__main__":
    main()