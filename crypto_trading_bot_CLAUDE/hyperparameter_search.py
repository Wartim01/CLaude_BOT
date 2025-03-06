#!/usr/bin/env python
# hyperparameter_search.py
"""
Script pour rechercher automatiquement les meilleurs hyperparamètres 
pour le modèle LSTM de prédiction de crypto-monnaies
"""
import os
import sys
import json
import numpy as np
import pandas as pd
from datetime import datetime
from sklearn.model_selection import ParameterGrid
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping
import optuna
import joblib
import logging

# Ajouter le répertoire parent au PYTHONPATH
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from config.config import DATA_DIR, MODEL_CHECKPOINTS_DIR
from utils.logger import setup_logger
from ai.models.lstm_model import LSTMModel
from ai.models.feature_engineering import FeatureEngineering

# Configurer le logger
logger = setup_logger("hyperparameter_search")

class HyperparameterSearch:
    """
    Classe pour la recherche des meilleurs hyperparamètres pour le modèle LSTM
    """
    def __init__(self, data_path, symbol, timeframe, search_method='optuna'):
        """
        Initialise la recherche d'hyperparamètres
        
        Args:
            data_path: Chemin vers les données
            symbol: Symbole de trading (ex: BTCUSDT)
            timeframe: Intervalle de temps (ex: 15m)
            search_method: Méthode de recherche ('grid', 'random', 'optuna')
        """
        self.data_path = data_path
        self.symbol = symbol
        self.timeframe = timeframe
        self.search_method = search_method
        self.feature_engineering = FeatureEngineering(save_scalers=True)
        
        # Résultats de la recherche
        self.results = []
        self.best_params = None
        
        # Répertoire pour sauvegarder les résultats
        self.output_dir = os.path.join(DATA_DIR, "hyperparameter_search")
        os.makedirs(self.output_dir, exist_ok=True)
        
    def load_data(self):
        """
        Charge et prépare les données
        
        Returns:
            Les données préparées
        """
        # Déterminer le chemin du fichier
        file_path = os.path.join(self.data_path, f"{self.symbol}_{self.timeframe}.csv")
        
        # Vérifier si le fichier existe
        if not os.path.exists(file_path):
            # Chercher un fichier qui contient le symbole et le timeframe
            possible_files = [f for f in os.listdir(self.data_path) if f.endswith('.csv') and self.symbol in f and self.timeframe in f]
            
            if possible_files:
                file_path = os.path.join(self.data_path, possible_files[0])
                logger.info(f"Utilisation du fichier trouvé: {file_path}")
            else:
                raise FileNotFoundError(f"Aucun fichier de données trouvé pour {self.symbol}_{self.timeframe} dans {self.data_path}")
        
        # Charger les données
        logger.info(f"Chargement des données depuis {file_path}...")
        df = pd.read_csv(file_path)
        
        # Convertir timestamp en datetime et définir comme index
        if 'timestamp' in df.columns:
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df.set_index('timestamp', inplace=True)
        elif 'date' in df.columns:
            df['timestamp'] = pd.to_datetime(df['date'])
            df.set_index('timestamp', inplace=True)
        
        # Trier par index pour assurer l'ordre chronologique
        df = df.sort_index()
        
        logger.info(f"Données chargées: {len(df)} lignes de {df.index.min()} à {df.index.max()}")
        
        # 1. Préparer les données
        featured_data = self.feature_engineering.create_features(
            df, 
            include_time_features=True,
            include_price_patterns=True
        )
        
        # Normaliser les données
        normalized_data = self.feature_engineering.scale_features(
            featured_data,
            is_training=True,
            method='standard',
            feature_group='lstm'
        )
        
        # Diviser en ensembles d'entraînement et de validation (chronologiquement)
        train_size = int(len(normalized_data) * 0.7)
        val_size = int(len(normalized_data) * 0.15)
        
        train_data = normalized_data.iloc[:train_size]
        val_data = normalized_data.iloc[train_size:train_size+val_size]
        test_data = normalized_data.iloc[train_size+val_size:]
        
        return train_data, val_data, test_data
    
    def prepare_sequences(self, data, sequence_length, horizons):
        """
        Prépare les séquences d'entrée et les cibles
        
        Args:
            data: DataFrame avec les données normalisées
            sequence_length: Longueur des séquences
            horizons: Horizons de prédiction
            
        Returns:
            Séquences X et cibles y
        """
        X, y = self.feature_engineering.create_multi_horizon_data(
            data, 
            sequence_length=sequence_length,
            horizons=horizons,
            is_training=True
        )
        
        return X, y
    
    def objective(self, trial):
        """
        Fonction objectif pour Optuna
        
        Args:
            trial: Objet trial d'Optuna
            
        Returns:
            Score d'évaluation du modèle (à minimiser)
        """
        # Charger les données
        train_data, val_data, _ = self.load_data()
        
        # Définir les hyperparamètres à optimiser
        lstm_units = []
        for i in range(trial.suggest_int('n_layers', 1, 3)):
            lstm_units.append(trial.suggest_int(f'lstm_units_{i}', 32, 256))
        
        dropout_rate = trial.suggest_float('dropout_rate', 0.1, 0.5)
        learning_rate = trial.suggest_float('learning_rate', 1e-5, 1e-2, log=True)
        sequence_length = trial.suggest_categorical('sequence_length', [30, 60, 90, 120])
        batch_size = trial.suggest_categorical('batch_size', [32, 64, 128])
        
        # Utiliser des horizons courts pour une optimisation plus rapide
        horizons = [12, 24]  # Optimisés pour les horizons qui posent problème
        
        # Préparer les séquences
        X_train, y_train = self.prepare_sequences(train_data, sequence_length, horizons)
        X_val, y_val = self.prepare_sequences(val_data, sequence_length, horizons)
        
        # Créer le modèle
        model = LSTMModel(
            input_length=sequence_length,
            feature_dim=X_train.shape[2],
            lstm_units=lstm_units,
            dropout_rate=dropout_rate,
            learning_rate=learning_rate,
            prediction_horizons=horizons,
            use_attention=trial.suggest_categorical('use_attention', [True, False])
        )
        
        # Callbacks
        callbacks = [
            EarlyStopping(
                monitor='val_loss',
                patience=10,
                restore_best_weights=True
            )
        ]
        
        # Entraîner le modèle
        history = model.model.fit(
            x=X_train,
            y=y_train,
            epochs=50,  # Limiter pour l'optimisation
            batch_size=batch_size,
            validation_data=(X_val, y_val),
            callbacks=callbacks,
            verbose=0
        )
        
        # Évaluer le modèle
        val_loss = history.history['val_loss'][-1]
        
        # Calculer des métriques spécifiques de direction pour chaque horizon
        direction_accuracies = []
        
        predictions = model.model.predict(X_val)
        
        for i, horizon_idx in enumerate(range(len(horizons))):
            y_true = y_val[horizon_idx].flatten()
            y_pred = (predictions[horizon_idx].flatten() > 0.5).astype(int)
            
            # Calculer l'accuracy directionnelle
            accuracy = np.mean(y_true == y_pred)
            direction_accuracies.append(accuracy)
        
        # Mettre plus de poids sur les horizons faibles (24 et 96)
        # Horizon 12 est le premier, horizon 24 est le second
        weighted_accuracy = direction_accuracies[0] * 0.3 + direction_accuracies[1] * 0.7
        
        # Objectif: maximiser l'accuracy pondérée (Optuna minimise par défaut)
        return -weighted_accuracy  # Négation pour maximiser
    
    def run_optuna_search(self, n_trials=100):
        """
        Lance une recherche d'hyperparamètres avec Optuna
        
        Args:
            n_trials: Nombre d'essais
            
        Returns:
            Les meilleurs paramètres trouvés
        """
        study = optuna.create_study(direction='minimize')
        study.optimize(self.objective, n_trials=n_trials)
        
        # Meilleurs paramètres
        best_params = study.best_params
        best_value = -study.best_value  # Convertir à l'accuracy (positive)
        
        # Reconstruire les unités LSTM en une liste
        lstm_units = []
        for i in range(best_params['n_layers']):
            lstm_units.append(best_params[f'lstm_units_{i}'])
            del best_params[f'lstm_units_{i}']
        
        del best_params['n_layers']
        best_params['lstm_units'] = lstm_units
        
        logger.info(f"Meilleurs paramètres: {best_params}, Accuracy: {best_value:.4f}")
        
        self.best_params = best_params
        
        # Sauvegarder les résultats
        results_path = os.path.join(self.output_dir, f"{self.symbol}_{self.timeframe}_optuna_results.json")
        
        with open(results_path, 'w') as f:
            json.dump({
                'best_params': best_params,
                'best_value': best_value,
                'timestamp': datetime.now().isoformat()
            }, f, indent=2)
        
        # Sauvegarder l'étude complète
        study_path = os.path.join(self.output_dir, f"{self.symbol}_{self.timeframe}_optuna_study.pkl")
        joblib.dump(study, study_path)
        
        return best_params
    
    def find_best_parameters(self):
        """
        Exécute la recherche d'hyperparamètres selon la méthode choisie
        
        Returns:
            Les meilleurs paramètres trouvés
        """
        if self.search_method == 'optuna':
            return self.run_optuna_search()
        else:
            logger.error(f"Méthode de recherche non supportée: {self.search_method}")
            return None

def run_hyperparameter_search(data_path, symbol, timeframe, method='optuna', n_trials=50):
    """
    Fonction principale pour lancer la recherche d'hyperparamètres
    
    Args:
        data_path: Chemin vers les données
        symbol: Symbole de trading
        timeframe: Intervalle de temps
        method: Méthode de recherche
        n_trials: Nombre d'essais pour les méthodes basées sur l'optimisation bayésienne
        
    Returns:
        Les meilleurs paramètres trouvés
    """
    logger.info(f"Début de la recherche d'hyperparamètres pour {symbol} {timeframe}")
    
    searcher = HyperparameterSearch(
        data_path=data_path,
        symbol=symbol,
        timeframe=timeframe,
        search_method=method
    )
    
    # Vérifier si des résultats existent déjà
    results_path = os.path.join(searcher.output_dir, f"{symbol}_{timeframe}_optuna_results.json")
    
    if os.path.exists(results_path):
        with open(results_path, 'r') as f:
            results = json.load(f)
        logger.info(f"Résultats précédents trouvés: {results['best_params']}")
        
        # Option de réutiliser les résultats précédents ou relancer la recherche
        best_params = results['best_params']
    else:
        # Lancer une nouvelle recherche
        best_params = searcher.find_best_parameters()
    
    return best_params

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Recherche d'hyperparamètres pour le modèle LSTM")
    
    parser.add_argument("--symbol", type=str, required=True,
                      help="Symbole de la paire de trading (ex: BTCUSDT)")
    parser.add_argument("--timeframe", type=str, required=True,
                      help="Intervalle de temps (ex: 15m, 1h)")
    parser.add_argument("--data_path", type=str, required=True,
                      help="Chemin vers les données de marché")
    parser.add_argument("--method", type=str, default="optuna",
                      help="Méthode de recherche (grid, random, optuna)")
    parser.add_argument("--n_trials", type=int, default=50,
                      help="Nombre d'essais pour les méthodes basées sur l'optimisation bayésienne")
    
    args = parser.parse_args()
    
    best_params = run_hyperparameter_search(
        data_path=args.data_path,
        symbol=args.symbol,
        timeframe=args.timeframe,
        method=args.method,
        n_trials=args.n_trials
    )
    
    print("Meilleurs paramètres trouvés:")
    print(json.dumps(best_params, indent=2))