from utils.logger import setup_logger
logger = setup_logger("model_trainer")
# ai/models/model_trainer.py
"""
Module d'entraînement du modèle LSTM avec validation croisée temporelle
"""
import os
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Union, Optional
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau, Callback
import json

from ai.models.lstm_model import LSTMModel
from ai.models.feature_engineering import FeatureEngineering
from config.config import DATA_DIR
from utils.visualizer import TradeVisualizer
from config.model_params import LSTM_DEFAULT_PARAMS

# Configuration des GPU pour TensorFlow - S'exécute une seule fois au démarrage
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        logical_gpus = tf.config.experimental.list_logical_devices('GPU')
        logger.info(f"{len(gpus)} GPU(s) physiques et {len(logical_gpus)} GPU(s) logiques détectés - Memory growth activé")
    except RuntimeError as e:
        logger.error(f"Erreur lors de la configuration GPU: {e}")
else:
    logger.info("Aucun GPU détecté. Exécution sur CPU.")

class EarlyStoppingOnMemoryLeak(Callback):
    """
    Callback pour détecter et arrêter l'entraînement en cas de fuite mémoire
    """
    def __init__(self, memory_threshold_mb: float = 1000):
        super().__init__()
        self.memory_threshold_mb = memory_threshold_mb
        self.starting_memory = 0
        
    def on_train_begin(self, logs=None):
        # Mesurer l'utilisation mémoire au début
        self.starting_memory = self._get_memory_usage()
        logger.info(f"Mémoire au début de l'entraînement: {self.starting_memory:.2f} MB")
    
    def on_epoch_end(self, epoch, logs=None):
        # Vérifier l'utilisation mémoire à la fin de chaque époque
        current_memory = self._get_memory_usage()
        memory_increase = current_memory - self.starting_memory
        
        if memory_increase > self.memory_threshold_mb:
            logger.warning(f"Fuite mémoire détectée ! Augmentation: {memory_increase:.2f} MB. Arrêt de l'entraînement.")
            self.model.stop_training = True
    
    def _get_memory_usage(self):
        """Mesure la consommation mémoire actuelle en MB"""
        import psutil
        process = psutil.Process(os.getpid())
        memory_info = process.memory_info()
        return memory_info.rss / (1024 * 1024)  # Convertir en MB

class ConceptDriftDetector(Callback):
    """
    Callback pour détecter la dérive conceptuelle pendant l'entraînement
    """
    def __init__(self, validation_data, threshold=0.15, patience=3):
        super().__init__()
        self.validation_data = validation_data
        self.threshold = threshold
        self.patience = patience
        self.baseline_metrics = None
        self.degradation_count = 0
        
    def on_train_begin(self, logs=None):
        # Évaluer le modèle sur les données de validation pour établir une référence
        metrics = self.model.evaluate(
            self.validation_data[0], 
            self.validation_data[1], 
            verbose=0
        )
        self.baseline_metrics = metrics[0]  # Prendre la loss globale comme référence
        logger.info(f"Métrique de référence établie: {self.baseline_metrics:.4f}")
    
    def on_epoch_end(self, epoch, logs=None):
        # Évaluer le modèle sur les données de validation
        current_metrics = self.model.evaluate(
            self.validation_data[0], 
            self.validation_data[1], 
            verbose=0
        )[0]
        
        # Calculer la dégradation relative
        degradation = (current_metrics - self.baseline_metrics) / self.baseline_metrics
        
        logger.debug(f"Epoch {epoch+1}: Dégradation des métriques: {degradation:.4f}")
        
        # Vérifier la dérive conceptuelle
        if degradation > self.threshold:
            self.degradation_count += 1
            logger.warning(f"Dérive conceptuelle possible détectée, compteur: {self.degradation_count}/{self.patience}")
            
            if self.degradation_count >= self.patience:
                logger.warning("Dérive conceptuelle confirmée. Arrêt de l'entraînement.")
                self.model.stop_training = True
        else:
            # Réinitialiser le compteur si la dégradation est en dessous du seuil
            self.degradation_count = 0
            # Mettre à jour la référence si les performances s'améliorent
            if current_metrics < self.baseline_metrics:
                self.baseline_metrics = current_metrics
                logger.info(f"Nouvelle métrique de référence: {self.baseline_metrics:.4f}")

class ModelTrainer:
    """Class for training LSTM models with standardized pipeline"""
    
    def __init__(self, model_params=None, feature_engineering=None):
        """
        Initialize the trainer with model parameters
        
        Args:
            model_params: Dictionary of model parameters (uses defaults if None)
            feature_engineering: FeatureEngineering instance (creates a new one if None)
        """
        # If no params provided, use the default params from config
        if model_params is None:
            model_params = LSTM_DEFAULT_PARAMS
            logger.info("Using default model parameters from config")
        else:
            # Fill in missing parameters with defaults
            for key, value in LSTM_DEFAULT_PARAMS.items():
                if key not in model_params:
                    model_params[key] = value
                    
        self.model_params = model_params
        self.feature_engineering = feature_engineering or FeatureEngineering(save_scalers=True)
        
        # Create the model with parameters
        self.model = LSTMModel(
            input_length=model_params.get('input_length', model_params.get('sequence_length', 60)),
            feature_dim=model_params.get('feature_dim', 30),
            lstm_units=model_params.get('lstm_units', [128, 64, 32]),
            dropout_rate=model_params.get('dropout_rate', 0.3),
            learning_rate=model_params.get('learning_rate', 0.001),
            use_attention=model_params.get('use_attention', True),
            l1_reg=model_params.get('l1_reg', model_params.get('l1_regularization', 0.0001)),
            l2_reg=model_params.get('l2_reg', model_params.get('l2_regularization', 0.0001)),
            prediction_horizons=model_params.get('prediction_horizons', [4])  # Default to 1h with 15m data
        )
        
        logger.info(f"Created model with parameters from config/optimization: {model_params}")
    
    def train(self, X_train, y_train, epochs=100, batch_size=32, verbose=1, validation_data=None, callbacks=None, class_weight=None):
        """
        Train the model on the provided data
        
        Args:
            X_train: Training data features
            y_train: Training data labels
            epochs: Number of epochs
            batch_size: Batch size
            verbose: Verbosity mode
            validation_data: Validation data (features, labels)
            callbacks: List of Keras callbacks (creates default if None)
            class_weight: Dictionary mapping class indices to a weight for the class
            
        Returns:
            Training history
        """
        # Create default callbacks if none provided
        if callbacks is None:
            callbacks = self._get_default_callbacks()
            
        # Ajouter un callback pour afficher les métriques détaillées à la fin de chaque époque
        from tensorflow.keras.callbacks import Callback
        class DetailedMetricLogger(Callback):
            def on_epoch_end(self, epoch, logs=None):
                logs = logs or {}
                metrics_str = ' - '.join([f"{k}: {v:.4f}" for k, v in logs.items()])
                # Replace print with logger.info() for unified logging of epoch metrics
                logger.info(f"Époque {epoch+1}/{self.params['epochs']} - {metrics_str}")
                
        callbacks.append(DetailedMetricLogger())
        
        # Train the model with verbose=2 for detailed metrics per epoch
        history = self.model.model.fit(
            X_train, y_train,
            epochs=epochs,
            batch_size=batch_size,
            verbose=verbose,
            validation_data=validation_data,  # new parameter passed here
            callbacks=callbacks,
            class_weight=class_weight
        )
        
        return history
        
    def prepare_data(self, data, is_training=True):
        """
        Prepare data for model training or prediction
        
        Args:
            data: DataFrame with raw data
            is_training: Whether this is for training
            
        Returns:
            X, y (if is_training=True) or X (if is_training=False)
        """
        # Create features
        featured_data = self.feature_engineering.create_features(data)
        
        # Normalize features
        normalized_data = self.feature_engineering.scale_features(
            featured_data,
            is_training=is_training
        )
        
        # Create sequences
        horizons = self.model_params.get('prediction_horizons', [4])
        result = self.feature_engineering.create_multi_horizon_data(
            normalized_data,
            sequence_length=self.model_params.get('input_length', 60),
            horizons=horizons,
            is_training=is_training
        )
        
        # Check feature dimensions for debugging
        if is_training and isinstance(result, tuple) and len(result) >= 1:
            X = result[0]
            if isinstance(X, np.ndarray) and len(X.shape) == 3:
                logger.info(f"Generated training data with shape: {X.shape}, features: {X.shape[2]}")
                
                # Update model params with actual feature dimension
                if self.model_params.get('feature_dim', 0) != X.shape[2]:
                    logger.info(f"Updating model feature dimension from {self.model_params.get('feature_dim')} to {X.shape[2]}")
                    self.model_params['feature_dim'] = X.shape[2]
        
        return result
    
    def _get_default_callbacks(self):
        """
        Get default training callbacks using parameters from config
        
        Returns:
            List of callbacks
        """
        return [
            EarlyStopping(
                monitor='val_loss',
                patience=self.model_params.get('early_stopping_patience', LSTM_DEFAULT_PARAMS['early_stopping_patience']),
                restore_best_weights=True
            ),
            ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=self.model_params.get('reduce_lr_patience', LSTM_DEFAULT_PARAMS['reduce_lr_patience']),
                min_lr=1e-6
            ),
            EarlyStoppingOnMemoryLeak(memory_threshold_mb=1000)  # appel mis à jour
        ]