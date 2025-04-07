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
from tqdm import tqdm  # For progress indication
import logging

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
logger.setLevel(logging.DEBUG)

def update_model_params_file(best_params, timeframe, f1_score):
    import json, os
    from datetime import datetime
    
    # Chemin du fichier de paramètres optimisés
    params_path = os.path.join("config", "optimized_params.json")
    
    # Créer le répertoire si nécessaire
    os.makedirs(os.path.dirname(params_path), exist_ok=True)
    
    # Charger les paramètres existants s'ils existent
    existing_params = {}
    if os.path.exists(params_path):
        try:
            with open(params_path, 'r') as f:
                existing_params = json.load(f)
            logger.debug(f"Paramètres existants chargés depuis {params_path}")
        except Exception as e:
            logger.warning(f"Erreur lors du chargement des paramètres existants: {e}")
    
    # Préparer les nouveaux paramètres
    updated_params = best_params.copy()
    updated_params["last_optimized"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    updated_params["f1_score"] = f1_score
    
    # Mettre à jour uniquement le timeframe spécifié, préserver les autres
    existing_params[timeframe] = updated_params
    
    # Enregistrer les paramètres mis à jour
    try:
        with open(params_path, 'w') as f:
            json.dump(existing_params, f, indent=2)
        logger.info(f"Paramètres optimisés mis à jour pour {timeframe} avec F1={f1_score:.4f}")
        logger.debug(f"Détail des paramètres enregistrés pour {timeframe}: {json.dumps(updated_params)}")
    except Exception as e:
        logger.error(f"Erreur lors de l'enregistrement des paramètres optimisés: {e}")

class BestModelCallback:
    """
    Callback for Optuna to update model_params.py when a new best trial is found
    """
    def __init__(self, timeframe):
        self.best_value = -float('inf')
        self.best_params = None
        self.timeframe = timeframe
        logger.info(f"BestModelCallback initialisé pour timeframe: {timeframe}")
    
    def __call__(self, study, trial):
        logger.debug(f"Callback __call__ invoqué pour l'essai {trial.number}")
        
        # Ajouter une gestion d'erreur pour vérifier s'il y a des essais terminés
        try:
            # Ne vérifier que si l'essai actuel est terminé
            if trial.state == optuna.trial.TrialState.COMPLETE:
                # Obtenir les essais terminés
                completed_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
                
                logger.debug(f"Essais terminés: {len(completed_trials)}")
                
                # Ne procéder que s'il y a au moins un essai terminé
                if completed_trials:
                    # Trouver le meilleur essai manuellement parmi les essais terminés
                    best_trial = max(completed_trials, key=lambda t: t.value if t.value is not None else float('-inf'))
                    
                    logger.debug(f"Meilleur essai actuel: #{best_trial.number} avec valeur={best_trial.value}")
                    logger.debug(f"Meilleure valeur précédente: {self.best_value}")
                    
                    # Vérifier si nous avons trouvé un nouvel essai meilleur
                    if best_trial.value is not None and best_trial.value > self.best_value:
                        old_value = self.best_value
                        self.best_value = best_trial.value
                        self.best_params = best_trial.params
                        
                        logger.info(f"Nouveau meilleur essai trouvé! Essai {best_trial.number}: F1={self.best_value:.4f} (précédent: {old_value:.4f})")
                        
                        # Mettre à jour le fichier de paramètres avec les nouveaux meilleurs paramètres
                        update_model_params_file(self.best_params, self.timeframe, self.best_value)
                        logger.info(f"Fichier de configuration mis à jour avec les paramètres de l'essai {best_trial.number}")
                    else:
                        logger.debug(f"Pas de nouveau meilleur essai trouvé (meilleur reste: F1={self.best_value:.4f})")
        except Exception as e:
            # Cela peut se produire s'il y a des problèmes avec l'accès aux essais ou aux valeurs
            logger.warning(f"Impossible de mettre à jour les paramètres après l'essai {trial.number}: {str(e)}")
            import traceback
            logger.debug(traceback.format_exc())

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
        # Set a fixed seed for reproducibility so that the train/validation split remains constant
        import numpy as np
        import tensorflow as tf
        np.random.seed(42)
        tf.random.set_seed(42)
        
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
        logger.info("Préparation des données fixées pour l'optimisation (ensembles d'entraînement et de validation constants)")
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
        # Assurer la reproductibilité pour chaque essai
        np.random.seed(42)
        tf.random.set_seed(42)
        
        logger.debug(f"Démarrage de l'essai #{trial.number}")
        
        try:
            # Exemple de suggestion pour les unités LSTM
            lstm_units_first = trial.suggest_int("lstm_units_first", 136, 256)  # minimum fixé à 136
            lstm_layers = trial.suggest_int("lstm_layers", 2, 3)  # Limiter à 2 ou 3 couches
            
            logger.debug(f"Essai #{trial.number}: Couches LSTM: {lstm_layers}, Unités première couche: {lstm_units_first}")
            
            if lstm_layers == 2:
                lstm_units = [lstm_units_first, lstm_units_first // 2]
            else:  # lstm_layers == 3
                lstm_units = [lstm_units_first, lstm_units_first // 2, lstm_units_first // 2]
            
            dropout_rate = trial.suggest_float("dropout_rate", 0.2, 0.5)
            learning_rate = trial.suggest_float("learning_rate", 1e-5, 1e-3, log=True)
            
            logger.debug(f"Essai #{trial.number}: Dropout: {dropout_rate}, Learning rate: {learning_rate}")
            
            # Taille du batch parmi des valeurs fixes
            batch_size = trial.suggest_categorical("batch_size", [16, 32, 64, 128])
            logger.debug(f"Essai #{trial.number}: Batch size: {batch_size}")
            
            # Fixed sequence_length = 60
            sequence_length = 60
            logger.debug(f"Essai #{trial.number}: Longueur de séquence fixée à {sequence_length}")
            
            # Process a small batch of data to determine the actual feature dimension
            logger.debug(f"Essai #{trial.number}: Préparation d'un échantillon de données pour déterminer la dimension des caractéristiques")
            X_sample, _ = self.feature_engineering.create_multi_horizon_data(
                self.normalized_train.iloc[:100],
                sequence_length=sequence_length,
                horizons=[4],
                is_training=True
            )
            
            # Check if X_sample has the expected 3D shape
            if X_sample is None or len(X_sample) == 0:
                logger.error(f"Essai #{trial.number}: Échec de création des séquences d'entrée - X_sample est vide")
                return -1.0
                
            if len(X_sample.shape) != 3:
                logger.error(f"Essai #{trial.number}: X_sample a une forme inattendue: {X_sample.shape}, tensor 3D attendu")
                return -1.0
            
            # Get the actual feature dimension from the processed data
            actual_feature_dim = X_sample.shape[2]
            logger.debug(f"Essai #{trial.number}: Dimension de caractéristiques détectée: {actual_feature_dim}")
            
            # Use a fixed feature dimension instead of a dynamic one
            FIXED_FEATURE_DIM = 67  # Hardcode this to ensure consistency
            logger.debug(f"Essai #{trial.number}: Utilisation d'une dimension fixe de caractéristiques: {FIXED_FEATURE_DIM}")
            
            # Suggestion des paramètres de régularisation
            l1_reg = trial.suggest_float("l1_reg", 1e-6, 1e-3, log=True)
            l2_reg = trial.suggest_float("l2_reg", 1e-6, 1e-3, log=True)
            logger.debug(f"Essai #{trial.number}: Régularisation L1: {l1_reg}, L2: {l2_reg}")
            
            # Build model parameters with FIXED feature dimensions to avoid None shape issues
            model_params = {
                "input_length": sequence_length,  # Fixed to 60
                "feature_dim": FIXED_FEATURE_DIM,  # Fixed dimension
                "lstm_units": lstm_units,
                "dropout_rate": dropout_rate,
                "learning_rate": learning_rate,
                "l1_reg": l1_reg,
                "l2_reg": l2_reg,
                "use_attention": False,
                "use_residual": False,
                "prediction_horizons": [(4, "1h", True)]
            }
            
            logger.debug(f"Essai #{trial.number}: Paramètres du modèle configurés: {json.dumps(model_params, indent=2)}")
            
            # Create trainer with explicit error handling for shape issues
            try:
                logger.debug(f"Essai #{trial.number}: Création du ModelTrainer")
                trainer = ModelTrainer(model_params)
                
                # Try to build the model with explicit error handling
                try:
                    logger.debug(f"Essai #{trial.number}: Construction du modèle à sortie unique")
                    # Use eager execution temporarily to help with shape issues
                    tf.config.run_functions_eagerly(True)
                    single_model = trainer.model.build_single_output_model(horizon_idx=0)
                    # Return to normal execution mode after model creation
                    tf.config.run_functions_eagerly(False)
                    logger.debug(f"Essai #{trial.number}: Modèle construit avec succès")
                except ValueError as e:
                    if "Shapes used to initialize variables" in str(e):
                        logger.error(f"Essai #{trial.number}: Erreur d'initialisation des formes: {str(e)}")
                        return -1.0
                    raise
                
                # Rest of the training and evaluation code...
                # Assurer la reproductibilité pour chaque essai
                np.random.seed(42)
                tf.random.set_seed(42)
                
                try:
                    # Configuration des paramètres avec les nouvelles plages recommandées
                    lstm_units_first = trial.suggest_int("lstm_units_first", 64, 256)
                    lstm_layers = trial.suggest_int("lstm_layers", 1, 3)
                    lstm_units = [lstm_units_first]
                    for i in range(1, lstm_layers):
                        lstm_units.append(lstm_units_first // 2)
                    
                    dropout_rate = trial.suggest_float("dropout_rate", 0.2, 0.5)
                    learning_rate = trial.suggest_float("learning_rate", 1e-5, 1e-3, log=True)
                    
                    # Taille du batch parmi des valeurs fixes
                    batch_size = trial.suggest_categorical("batch_size", [16, 32, 64, 128])
                    
                    # IMPORTANT: Hardcode sequence_length to 60, do NOT let Optuna suggest it
                    sequence_length = 60
                    
                    # Process a small batch of data to determine the actual feature dimension
                    X_sample, _ = self.feature_engineering.create_multi_horizon_data(
                        self.normalized_train.iloc[:100],
                        sequence_length=sequence_length,
                        horizons=[4],
                        is_training=True
                    )
                    
                    # Check if X_sample has the expected 3D shape
                    if X_sample is None or len(X_sample) == 0:
                        logger.error("Failed to create input sequences - X_sample is empty")
                        raise ValueError("Failed to create valid input sequences")
                        
                    if len(X_sample.shape) != 3:
                        logger.error(f"X_sample has unexpected shape: {X_sample.shape}, expected 3D tensor")
                        raise ValueError(f"Expected 3D tensor, got shape {X_sample.shape}")
                    
                    # Get the actual feature dimension from the processed data
                    actual_feature_dim = X_sample.shape[2]
                    logger.info(f"Detected actual feature dimension: {actual_feature_dim}")
                    
                    # Ensure feature dimension is fixed and known before building the model
                    # This is critical to avoid the "None" dimension error
                    if actual_feature_dim <= 0:
                        logger.error(f"Invalid feature dimension: {actual_feature_dim}")
                        raise ValueError(f"Feature dimension must be positive, got {actual_feature_dim}")
                    
                    # Build model parameters - be explicit about all dimensions
                    model_params = {
                        "input_length": sequence_length,  # Fixed to 60
                        "feature_dim": actual_feature_dim,  # Use the detected feature dimension
                        "lstm_units": lstm_units,
                        "dropout_rate": dropout_rate,
                        "learning_rate": learning_rate,
                        "l1_reg": trial.suggest_float("l1_reg", 1e-6, 1e-3, log=True),
                        "l2_reg": trial.suggest_float("l2_reg", 1e-6, 1e-3, log=True),
                        "use_attention": False,
                        "use_residual": False,
                        "prediction_horizons": [(4, "1h", True)]  # Use tuple format for consistency
                    }
                    
                    logger.info(f"Essai {trial.number}: {json.dumps(model_params, indent=2)}")
                    
                    # Créer le trainer avec les paramètres
                    trainer = ModelTrainer(model_params)
                    
                    # Créer un modèle à sortie unique pour l'optimisation
                    try:
                        single_model = trainer.model.build_single_output_model(horizon_idx=0)  # Utiliser l'horizon court terme
                    except ValueError as e:
                        if "Shapes used to initialize variables" in str(e):
                            logger.error(f"Shape error creating model: {str(e)}")
                            logger.error("This is likely due to undefined dimensions in the model inputs")
                            return -1.0
                        raise
                    
                    # Préparer les données pour ce modèle avec strong validation
                    X_train, y_lists = self.feature_engineering.create_multi_horizon_data(
                        self.normalized_train,
                        sequence_length=sequence_length,
                        horizons=[4],  # Use consistent horizons format
                        is_training=True
                    )
                    
                    # Debug information - log shape before any processing
                    logger.info(f"Original X_train shape: {X_train.shape}")
                    
                    # Verify X_train shape and reshape if needed
                    if len(X_train.shape) != 3:
                        logger.warning(f"X_train has unexpected shape: {X_train.shape}, attempting to reshape")
                        if len(X_train.shape) == 2:
                            # If X_train is 2D (samples, features), reshape to 3D (samples, sequence_length, features/sequence_length)
                            num_samples = X_train.shape[0]
                            total_features = X_train.shape[1]
                            
                            # Check if total_features is divisible by sequence_length
                            if total_features % sequence_length == 0:
                                feature_dim = total_features // sequence_length
                                X_train = X_train.reshape(num_samples, sequence_length, feature_dim)
                                logger.info(f"Reshaped X_train to {X_train.shape}")
                            else:
                                # If not divisible, we'll need to pad or truncate
                                logger.warning(f"Total features {total_features} not divisible by sequence length {sequence_length}")
                                # Create a new array with the right shape
                                feature_dim = actual_feature_dim  # Use the feature dimension we detected earlier
                                new_X_train = np.zeros((num_samples, sequence_length, feature_dim))
                                
                                # Fill with available data, truncating if necessary
                                for i in range(num_samples):
                                    flat_data = X_train[i]
                                    # Reshape the flat data into a 2D array of appropriate size
                                    if len(flat_data) >= sequence_length * feature_dim:
                                        reshaped_data = flat_data[:sequence_length * feature_dim].reshape(sequence_length, feature_dim)
                                    else:
                                        # Pad with zeros
                                        padded_data = np.zeros(sequence_length * feature_dim)
                                        padded_data[:len(flat_data)] = flat_data
                                        reshaped_data = padded_data.reshape(sequence_length, feature_dim)
                                    
                                    new_X_train[i] = reshaped_data
                                
                                X_train = new_X_train
                                logger.info(f"Created new X_train with shape {X_train.shape}")
                        else:
                            raise ValueError(f"Cannot reshape X_train with shape {X_train.shape}")
                    
                    # Verify timesteps dimension matches sequence_length
                    if X_train.shape[1] != sequence_length:
                        logger.error(f"X_train timesteps {X_train.shape[1]} doesn't match sequence_length {sequence_length}")
                        # Try to fix this by padding or truncating
                        if X_train.shape[1] < sequence_length:
                            # Pad with zeros
                            padding = np.zeros((X_train.shape[0], sequence_length - X_train.shape[1], X_train.shape[2]))
                            X_train = np.concatenate([X_train, padding], axis=1)
                            logger.info(f"Padded X_train to shape {X_train.shape}")
                        else:
                            # Truncate
                            X_train = X_train[:, :sequence_length, :]
                            logger.info(f"Truncated X_train to shape {X_train.shape}")
                    
                    # Also validate validation data with strong validation
                    X_val, y_val_lists = self.feature_engineering.create_multi_horizon_data(
                        self.normalized_val,
                        sequence_length=sequence_length,
                        horizons=[4],  # Use consistent horizons format
                        is_training=True
                    )
                    
                    # Debug information - log shape before any processing
                    logger.info(f"Original X_val shape: {X_val.shape}")
                    
                    # Verify X_val shape and reshape if needed - same logic as X_train
                    if len(X_val.shape) != 3:
                        logger.warning(f"X_val has unexpected shape: {X_val.shape}, attempting to reshape")
                        if len(X_val.shape) == 2:
                            # If X_val is 2D (samples, features), reshape to 3D (samples, sequence_length, features/sequence_length)
                            num_samples = X_val.shape[0]
                            total_features = X_val.shape[1]
                            
                            # Check if total_features is divisible by sequence_length
                            if total_features % sequence_length == 0:
                                feature_dim = total_features // sequence_length
                                X_val = X_val.reshape(num_samples, sequence_length, feature_dim)
                                logger.info(f"Reshaped X_val to {X_val.shape}")
                            else:
                                # If not divisible, we'll need to pad or truncate
                                feature_dim = actual_feature_dim  # Use the feature dimension we detected earlier
                                new_X_val = np.zeros((num_samples, sequence_length, feature_dim))
                                
                                # Fill with available data, truncating if necessary
                                for i in range(num_samples):
                                    flat_data = X_val[i]
                                    # Reshape the flat data into a 2D array of appropriate size
                                    if len(flat_data) >= sequence_length * feature_dim:
                                        reshaped_data = flat_data[:sequence_length * feature_dim].reshape(sequence_length, feature_dim)
                                    else:
                                        # Pad with zeros
                                        padded_data = np.zeros(sequence_length * feature_dim)
                                        padded_data[:len(flat_data)] = flat_data
                                        reshaped_data = padded_data.reshape(sequence_length, feature_dim)
                                    
                                    new_X_val[i] = reshaped_data
                                
                                X_val = new_X_val
                                logger.info(f"Created new X_val with shape {X_val.shape}")
                        else:
                            raise ValueError(f"Cannot reshape X_val with shape {X_val.shape}")
                    
                    # Verify X_val timesteps dimension matches sequence_length
                    if X_val.shape[1] != sequence_length:
                        logger.error(f"X_val timesteps {X_val.shape[1]} doesn't match sequence_length {sequence_length}")
                        # Try to fix this by padding or truncating
                        if X_val.shape[1] < sequence_length:
                            # Pad with zeros
                            padding = np.zeros((X_val.shape[0], sequence_length - X_val.shape[1], X_val.shape[2]))
                            X_val = np.concatenate([X_val, padding], axis=1)
                            logger.info(f"Padded X_val to shape {X_val.shape}")
                        else:
                            # Truncate
                            X_val = X_val[:, :sequence_length, :]
                            logger.info(f"Truncated X_val to shape {X_val.shape}")
                    
                    # Verify validation data shape matches training data shape
                    if X_train.shape[1:] != X_val.shape[1:]:
                        logger.error(f"Shape mismatch: X_train {X_train.shape[1:]}, X_val {X_val.shape[1:]}")
                        # Try to fix feature dimension mismatch
                        if X_train.shape[1] == X_val.shape[1] and X_train.shape[2] != X_val.shape[2]:
                            # Sequence length matches but feature dimension doesn't
                            min_features = min(X_train.shape[2], X_val.shape[2])
                            X_train = X_train[:, :, :min_features]
                            X_val = X_val[:, :, :min_features]
                            logger.info(f"Adjusted feature dimensions to match: X_train {X_train.shape}, X_val {X_val.shape}")
                        else:
                            raise ValueError("Training and validation data shapes don't match and cannot be fixed")
                    
                    # Extraire uniquement les directions pour l'horizon court terme
                    y_train = y_lists[0]  # Direction pour le premier horizon
                    y_val = y_val_lists[0]  # Direction pour le premier horizon
                    
                    # Vérification immédiate de la cohérence de la dimension temporelle
                    assert X_train.shape[1] == sequence_length, f"Mismatch: séquence attendue {sequence_length} vs données {X_train.shape[1]}"
                    
                    # Make sure y_train is the right shape (flattened to 1D if needed)
                    if len(y_train.shape) > 1:
                        y_train = y_train.flatten()
                        logger.info(f"Flattened y_train to shape {y_train.shape}")
                    
                    if len(y_val.shape) > 1:
                        y_val = y_val.flatten()
                        logger.info(f"Flattened y_val to shape {y_val.shape}")
                    
                    # Calculer les poids des classes pour l'équilibrage
                    class_counts = np.bincount(y_train.flatten().astype(int))
                    class_weights = {
                        0: len(y_train) / (2 * class_counts[0]) if class_counts[0] > 0 else 1.0,
                        1: len(y_train) / (2 * class_counts[1]) if class_counts[1] > 0 else 1.0
                    }
                    
                    # Build training callbacks: early stopping and learning rate reducer
                    from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau # type: ignore
                    callbacks = [
                        EarlyStopping(
                            monitor='val_loss',
                            patience=trial.suggest_int("early_stopping_patience", 5, 15),
                            restore_best_weights=True
                        ),
                        ReduceLROnPlateau(
                            monitor='val_loss',
                            factor=0.5,
                            patience=trial.suggest_int("reduce_lr_patience", 3, 7),
                            min_lr=1e-6,
                            verbose=1
                        )
                    ]
                    
                    # Log input shape to help diagnose any issues
                    logger.info(f"Training with input shape: {X_train.shape}, output shape: {y_train.shape}")
                    
                    # Perform a final check of the model input shape vs data shape
                    model_input_shape = single_model.input.shape
                    logger.info(f"Model expects input shape: {model_input_shape}")
                    if model_input_shape[1] != X_train.shape[1] or model_input_shape[2] != X_train.shape[2]:
                        logger.error(f"Model input shape {model_input_shape[1:]} doesn't match data shape {X_train.shape[1:]}")
                        # Try to rebuild the model with the correct shape
                        model_params["input_length"] = X_train.shape[1]
                        model_params["feature_dim"] = X_train.shape[2]
                        logger.info(f"Rebuilding model with adjusted parameters: input_length={model_params['input_length']}, feature_dim={model_params['feature_dim']}")
                        trainer = ModelTrainer(model_params)
                        single_model = trainer.model.build_single_output_model(horizon_idx=0)
                    
                    # Utilisation d'un sous-échantillon pour l'optimisation
                    max_samples = 5000
                    if len(X_train) > max_samples:
                        logger.info(f"Using {max_samples} samples for optimization (from {len(X_train)} total)")
                        X_train_sample = X_train[:max_samples]
                        y_train_sample = y_train[:max_samples]
                    else:
                        X_train_sample = X_train
                        y_train_sample = y_train
                        
                    # Entraîner le modèle sur un sous-ensemble de données pour réduire le temps d'optimisation
                    history = single_model.fit(
                        X_train_sample, y_train_sample,
                        validation_data=(X_val, y_val),
                        epochs=30,  # Epochs réduits pour l'optimisation
                        batch_size=batch_size,
                        class_weight=class_weights,
                        callbacks=callbacks,
                        verbose=0  # Réduire la verbosité pour optimiser
                    )
                    
                    # Récupérer les métriques
                    val_accuracy = max(history.history['val_accuracy'])
                    val_loss = min(history.history['val_loss'])
                    
                    # Évaluer sur l'ensemble de validation complet 
                    # Calculer le F1-score qui est plus approprié pour les données déséquilibrées
                    from sklearn.metrics import f1_score
                    y_pred = (single_model.predict(X_val) > 0.5).astype(int)
                    f1 = f1_score(y_val, y_pred)
                    
                    # Enregistrer les métriques dans l'historique des essais
                    trial_metrics = {
                        "val_accuracy": float(val_accuracy),
                        "val_loss": float(val_loss),
                        "f1_score": float(f1),
                        "params": model_params
                    }
                    
                    self.trials_history.append(trial_metrics)
                    self._save_trials_history()
                    
                    # Retourner le F1-score comme métrique d'optimisation
                    result = f1
                    tf.keras.backend.clear_session()  # Libère la mémoire GPU après chaque essai
                    trial.set_user_attr("completed", True)
                    return result
                    
                except Exception as e:
                    logger.error(f"Error in objective function: {str(e)}")
                    import traceback
                    logger.error(traceback.format_exc())
                    # Return a very low score so this trial is considered a failure
                    return -1.0  # Using pruned might prevent useful error messages
            
            except Exception as e:
                logger.error(f"Error creating model trainer: {str(e)}")
                return -1.0
                
        except Exception as e:
            logger.error(f"Error in objective function: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())
            return -1.0

    from sklearn.model_selection import StratifiedShuffleSplit
    from sklearn.metrics import f1_score
    import numpy as np

    def objective_stratified(self, trial):
        from sklearn.model_selection import StratifiedShuffleSplit
        from sklearn.metrics import f1_score
        import numpy as np

        def suggest_hyperparameters(trial, feature_dim):
            lstm_units_first = trial.suggest_int("lstm_units_first", 64, 256)
            lstm_layers = trial.suggest_int("lstm_layers", 1, 3)
            lstm_units = [lstm_units_first]
            for _ in range(1, lstm_layers):
                lstm_units.append(lstm_units_first // 2)
            return {
                "input_length": 60,
                "feature_dim": feature_dim,
                "lstm_units": lstm_units,
                "dropout_rate": trial.suggest_float("dropout_rate", 0.2, 0.5),
                "learning_rate": trial.suggest_float("learning_rate", 1e-5, 1e-3, log=True),
                "l1_reg": trial.suggest_float("l1_reg", 1e-6, 1e-3, log=True),
                "l2_reg": trial.suggest_float("l2_reg", 1e-6, 1e-3, log=True),
                "use_attention": False,
                "use_residual": False,
                "prediction_horizons": [(4, "1h", True)],
                "batch_size": trial.suggest_categorical("batch_size", [16, 32, 64, 128]),
                "sequence_length": 60,
                "l1_regularization": 6.358e-5,
                "l2_regularization": 0.001,
                "epochs": 30,
                "early_stopping_patience": trial.suggest_int("early_stopping_patience", 5, 15),
                "reduce_lr_patience": trial.suggest_int("reduce_lr_patience", 3, 7)
            }

        # ...existing code...
        X_all, y_lists = self.feature_engineering.create_multi_horizon_data(
            self.normalized_train,
            sequence_length=60,
            horizons=[4],
            is_training=True
        )
        y_all = y_lists[0].flatten()
        sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
        for train_idx, val_idx in sss.split(X_all, y_all):
            X_train, X_val = X_all[train_idx], X_all[val_idx]
            y_train, y_val = y_all[train_idx], y_all[val_idx]

        feature_dim = X_train.shape[2] if len(X_train.shape)==3 else X_train.shape[1]//60
        config = suggest_hyperparameters(trial, feature_dim)
        
        # Calculate class weights based on the training labels
        class_counts = np.bincount(y_train.flatten().astype(int))
        class_weights = {
            0: len(y_train) / (2 * class_counts[0]) if class_counts[0] > 0 else 1.0,
            1: len(y_train) / (2 * class_counts[1]) if class_counts[1] > 0 else 1.0
        }
        
        # Utiliser directement ModelTrainer pour construire et entraîner le modèle
        trainer = ModelTrainer(config)
        history = trainer.train(
            X_train, y_train,
            epochs=30,  # Fixed epochs value (30)
            batch_size=config["batch_size"],
            verbose=0,
            validation_data=(X_val, y_val),  # Added validation data for EarlyStopping and ReduceLROnPlateau
            class_weight=class_weights
        )
        model = trainer.model
        # --- existing code continues ---
        y_pred = (model.predict(X_val) > 0.5).astype(int)
        print("\n--- DEBUG ---")
        print("y_pred unique values:", np.unique(y_pred, return_counts=True))
        print("y_val unique values:", np.unique(y_val, return_counts=True))
        print("First preds vs true:", list(zip(y_pred[:10].flatten(), y_val[:10].flatten())))
        print("----------------\n")
        try:
            f1 = f1_score(y_val, y_pred, average='binary')
        except Exception as e:
            print("Erreur F1-score:", e)
            f1 = 0.0
        val_accuracy = max(history.history.get('val_accuracy', [0]))
        val_loss = min(history.history.get('val_loss', [np.inf]))
        trial.set_user_attr("val_accuracy", val_accuracy)
        trial.set_user_attr("val_loss", val_loss)
        trial.set_user_attr("f1_score", f1)
        trial.set_user_attr("params", config)
        tf.keras.backend.clear_session()  # Libère la mémoire GPU/CPU
        return f1

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
            logger.info(f"Utilisation de {len(callbacks)} callbacks fournis")
            for cb in callbacks:
                if isinstance(cb, BestModelCallback):
                    logger.info(f"BestModelCallback trouvé pour timeframe: {cb.timeframe}")
            optuna_callbacks.extend(callbacks)
        else:
            logger.warning("Aucun callback fourni - les meilleurs paramètres ne seront pas automatiquement sauvegardés")
        
        # Lancer l'optimisation
        try:
            study.optimize(self.objective_stratified, n_trials=n_trials, timeout=timeout, callbacks=optuna_callbacks)
            logger.info("Optimisation terminée avec succès")
        except Exception as e:
            logger.error(f"Erreur pendant l'optimisation: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())
        
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

def download_data_if_needed(symbol, timeframe, start_date=None, end_date=None):
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

def load_data(data_path, symbol, timeframe):
    import os
    import pandas as pd
    file_name = f"{symbol}_{timeframe}.csv"
    full_path = os.path.join(data_path, file_name)
    if not os.path.exists(full_path):
        raise FileNotFoundError(f"Data file not found: {full_path}")
    data = pd.read_csv(full_path)
    # Process data as needed (e.g. parse dates, set index)
    if 'timestamp' in data.columns:
        data['timestamp'] = pd.to_datetime(data['timestamp'])
        data.set_index('timestamp', inplace=True)
    return data

def main():
    """Point d'entrée principal"""
    parser = argparse.ArgumentParser(description="Optimisation des hyperparamètres du modèle LSTM")
    
    # Arguments pour les données
    parser.add_argument("--symbol", type=str, default="BTCUSDT", help="Paire de trading")
    parser.add_argument("--timeframe", type=str, default="15m", help="Intervalle de temps")
    
    # Support flexible argument naming (both hyphen and underscore)
    # parser.add_argument("--start-date", "--start_date", "--start", type=str, required=True, 
    #                   help="Date de début (YYYY-MM-DD)")
    # parser.add_argument("--end-date", "--end_date", "--end", type=str, required=True, 
    #                   help="Date de fin (YYYY-MM-DD)")
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
    parser.add_argument("--max_evals", type=int, default=100, help="Maximum evaluations for hyperparameter search")
    parser.add_argument("--output", type=str, required=True, help="Output path for best parameters")
    
    args = parser.parse_args()
    
    # Check for data availability
    if not download_data_if_needed(args.symbol, args.timeframe):
        logger.error("Impossible de continuer sans données")
        return
    
    # Charger les données avec les bons paramètres
    # load_data prend data_path, symbol et timeframe comme paramètres
    data_path = os.path.join(DATA_DIR, "market_data") if args.data_path == "data/market_data/" else args.data_path
    data = load_data(data_path, args.symbol, args.timeframe)
    
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
    cmd += f"--sequence-length 60 "  # Always force 60 regardless of what's in best_params
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
    logger.info("Une fois satisfaits, fixez ces hyperparamètres dans le fichier de configuration et redémarrez l’entraînement final pour vérifier l’impact sur l’accuracy.")

if __name__ == "__main__":
    main()

# Example command to run the script
# python hyperparameter_search.py --data_path "c:\Users\timot\OneDrive\Bureau\BOT TRADING BIG 2025\data" --symbol "BTCUSDT" --timeframe "15m" --max_evals 100 --output "c:\Users\timot\OneDrive\Bureau\BOT TRADING BIG 2025\crypto_trading_bot_CLAUDE\models\optimization\best_params.json"

