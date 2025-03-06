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
from datetime import datetime, timedelta
import json
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score

from ai.models.lstm_model import LSTMModel
from ai.models.feature_engineering import FeatureEngineering
from ai.models.model_trainer import ModelTrainer
from ai.models.model_validator import ModelValidator
from config.config import DATA_DIR
from utils.logger import setup_logger
from train_model import load_data, download_data_if_needed, load_historical_data

# Configuration du logger
logger = setup_logger("hyperparameter_search")

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
        Optimized objective function for Optuna with model caching and early pruning
        """
        try:
            # Configuration parameters with intelligent defaults based on data characteristics
            data_volatility = np.std(self.train_data['close'].pct_change().dropna())
            
            # Adjust parameter ranges based on data characteristics
            if data_volatility > 0.02:  # High volatility
                lstm_units_range = (96, 256)
                dropout_range = (0.3, 0.5)
            else:  # Lower volatility
                lstm_units_range = (64, 192)
                dropout_range = (0.2, 0.4)
            
            # Suggest hyperparameters
            lstm_units_first = trial.suggest_int("lstm_units_first", lstm_units_range[0], lstm_units_range[1])
            lstm_layers = trial.suggest_int("lstm_layers", 1, 3)
            dropout_rate = trial.suggest_float("dropout_rate", dropout_range[0], dropout_range[1])
            learning_rate = trial.suggest_float("learning_rate", 1e-5, 1e-2, log=True)
            
            # Efficient layering with decreasing units
            lstm_units = [lstm_units_first]
            for i in range(1, lstm_layers):
                lstm_units.append(lstm_units[-1] // 2)
            
            # Add advanced model architecture parameters
            use_attention = trial.suggest_categorical("use_attention", [False, True])  # Try both options
            batch_norm = trial.suggest_categorical("batch_norm", [True, False])
            
            # Model parameters
            model_params = {
                "input_length": trial.suggest_categorical("sequence_length", [30, 60, 90]),
                "feature_dim": self.normalized_train.shape[1] if hasattr(self.normalized_train, "shape") else 30,
                "lstm_units": lstm_units,
                "dropout_rate": dropout_rate,
                "learning_rate": learning_rate,
                "l1_reg": trial.suggest_float("l1_reg", 1e-6, 1e-3, log=True),
                "l2_reg": trial.suggest_float("l2_reg", 1e-6, 1e-3, log=True),
                "use_attention": use_attention,
                "use_residual": trial.suggest_categorical("use_residual", [False, True]),
                "prediction_horizons": [12, 24, 96],
                "use_batch_norm": batch_norm
            }
            
            # Log trial info
            logger.info(f"Trial {trial.number} - Testing: {json.dumps(model_params, indent=2)}")
            
            # Early pruning for obviously bad configurations
            if lstm_units_first > 200 and dropout_rate < 0.3:
                logger.info(f"Early pruning trial {trial.number}: high complexity with low regularization")
                raise optuna.exceptions.TrialPruned()
            
            # Create model with single output for optimization
            trainer = ModelTrainer(model_params)
            single_model = trainer.model.build_single_output_model(horizon_idx=0)  # Short-term direction
            
            # Prepare data efficiently - reuse for same sequence length
            sequence_length = model_params["input_length"]
            X_train, y_train_all = self._get_prepared_data(
                self.normalized_train, sequence_length, model_params["prediction_horizons"], True
            )
            
            X_val, y_val_all = self._get_prepared_data(
                self.normalized_val, sequence_length, model_params["prediction_horizons"], False
            )
            
            y_train = y_train_all[0]  # Direction for first horizon
            y_val = y_val_all[0]  # Direction for first horizon
            
            # Implement class balancing
            class_counts = np.bincount(y_train.flatten())
            if len(class_counts) > 1:
                class_weights = {
                    0: len(y_train) / (2 * class_counts[0]),
                    1: len(y_train) / (2 * class_counts[1])
                }
            else:
                class_weights = {0: 1.0, 1: 1.0}
            
            # Enhanced callbacks with reporting to Optuna for pruning
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
                ),
                optuna.integration.TFKerasPruningCallback(
                    trial, "val_accuracy", interval=1
                )
            ]
            
            # Train model with dynamic batch size
            batch_size = trial.suggest_categorical("batch_size", [16, 32, 64, 128])
            history = single_model.fit(
                x=X_train,
                y=y_train,
                epochs=30,
                batch_size=batch_size,
                validation_data=(X_val, y_val),
                class_weight=class_weights,
                callbacks=callbacks,
                verbose=0
            )
            
            # Evaluate with multiple metrics for better model selection
            val_loss, val_accuracy = single_model.evaluate(X_val, y_val, verbose=0)
            
            # Get predictions and calculate F1 score
            y_pred_proba = single_model.predict(X_val)
            y_pred = (y_pred_proba > 0.5).astype(int).flatten()
            f1 = f1_score(y_val, y_pred)
            
            # Save trial results
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
            
            # Save periodically
            if len(self.trials_history) % 5 == 0:
                self._save_trials_history()
            
            logger.info(f"Trial {trial.number}: F1={f1:.4f}, Accuracy={val_accuracy:.4f}")
            
            return f1  # Optimize F1 score
            
        except Exception as e:
            logger.error(f"Error during training: {str(e)}")
            return 0.0  # Default value in case of error
    
    def _save_trials_history(self):
        """Sauvegarde l'historique des essais"""
        history_path = os.path.join(self.output_dir, f"trials_history_{datetime.now().strftime('%Y%m%d')}.json")
        
        try:
            with open(history_path, 'w') as f:
                json.dump(self.trials_history, f, indent=2, default=str)
            logger.info(f"Historique des essais sauvegardé: {history_path}")
        except Exception as e:
            logger.error(f"Erreur lors de la sauvegarde de l'historique: {str(e)}")
    
    def run_optimization(self, n_trials=50, timeout=None):
        """
        Lance l'optimisation des hyperparamètres
        
        Args:
            n_trials: Nombre d'essais
            timeout: Temps maximum en secondes
            
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
        
        # Lancer l'optimisation
        study.optimize(self.objective, n_trials=n_trials, timeout=timeout)
        
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


def main():
    """Point d'entrée principal"""
    parser = argparse.ArgumentParser(description="Optimisation des hyperparamètres du modèle LSTM")
    
    # Arguments pour les données
    parser.add_argument("--symbol", type=str, default="BTCUSDT", help="Paire de trading")
    parser.add_argument("--timeframe", type=str, default="15m", help="Intervalle de temps")
    parser.add_argument("--start-date", type=str, required=True, help="Date de début (YYYY-MM-DD)")
    parser.add_argument("--end-date", type=str, required=True, help="Date de fin (YYYY-MM-DD)")
    
    # Arguments pour l'optimisation
    parser.add_argument("--n-trials", type=int, default=50, help="Nombre d'essais")
    parser.add_argument("--timeout", type=int, default=None, help="Temps maximum en secondes")
    parser.add_argument("--train-ratio", type=float, default=0.7, help="Ratio des données d'entraînement")
    parser.add_argument("--val-ratio", type=float, default=0.15, help="Ratio des données de validation")
    
    args = parser.parse_args()
    
    # Télécharger les données si nécessaire
    if not download_data_if_needed(args.symbol, args.timeframe, args.start_date, args.end_date):
        logger.error("Impossible de continuer sans données")
        return
    
    # Charger les données
    data = load_data(args.symbol, args.timeframe, args.start_date, args.end_date)
    
    if data.empty:
        logger.error("Données vides, impossible de continuer")
        return
    
    # Diviser les données en ensembles d'entraînement et de validation
    train_size = int(len(data) * args.train_ratio)
    val_size = int(len(data) * args.val_ratio)
    
    train_data = data.iloc[:train_size]
    val_data = data.iloc[train_size:train_size+val_size]
    
    logger.info(f"Données divisées: {len(train_data)} lignes pour l'entraînement, {len(val_data)} lignes pour la validation")
    
    # Créer l'optimiseur
    optimizer = LSTMHyperparameterOptimizer(
        train_data=train_data,
        val_data=val_data,
        symbol=args.symbol,
        timeframe=args.timeframe
    )
    
    # Lancer l'optimisation
    best_params = optimizer.run_optimization(
        n_trials=args.n_trials,
        timeout=args.timeout
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