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
from utils.logger import setup_logger
from utils.visualizer import TradeVisualizer

# Configuration des GPU pour TensorFlow
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        # Limiter la mémoire GPU utilisée
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        logical_gpus = tf.config.experimental.list_logical_devices('GPU')
        print(f"{len(gpus)} GPU(s) physiques et {len(logical_gpus)} GPU(s) logiques détectés")
    except RuntimeError as e:
        print(f"Erreur lors de la configuration GPU: {e}")

logger = setup_logger("model_trainer")

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
    """
    Classe pour l'entraînement et la validation du modèle LSTM
    """
    def __init__(self, model_params: Dict = None):
        """
        Initialise le ModelTrainer
        
        Args:
            model_params: Paramètres du modèle LSTM
        """
        # Paramètres par défaut du modèle
        default_params = {
            "input_length": 60,
            "feature_dim": 30,
            "lstm_units": [128, 64, 32],
            "dropout_rate": 0.3,
            "learning_rate": 0.001,
            "l1_reg": 0.0001,
            "l2_reg": 0.0001,
            "use_attention": True,
            "use_residual": True,
            "prediction_horizons": [12, 24, 96]  # 3h, 6h, 24h avec des bougies de 15min
        }
        
        # Fusionner avec les paramètres personnalisés
        self.model_params = {**default_params, **(model_params or {})}
        
        # Instancier le modèle
        self.model = LSTMModel(**self.model_params)
        
        # Instancier l'ingénieur de caractéristiques
        self.feature_engineering = FeatureEngineering(save_scalers=True)
        
        # Répertoires pour sauvegarder les résultats
        self.output_dir = os.path.join(DATA_DIR, "models")
        self.train_history_dir = os.path.join(self.output_dir, "training_history")
        
        # Créer les répertoires si nécessaire
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(self.train_history_dir, exist_ok=True)
        
        # Historique des entraînements
        self.training_history = []
    
    def prepare_data(self, data: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Prépare les données pour l'entraînement en créant des caractéristiques
        
        Args:
            data: DataFrame avec les données OHLCV brutes
            
        Returns:
            DataFrame avec les caractéristiques avancées
        """
        # Créer les caractéristiques
        featured_data = self.feature_engineering.create_features(
            data, 
            include_time_features=True,
            include_price_patterns=True
        )
        
        # Normaliser les caractéristiques
        normalized_data = self.feature_engineering.scale_features(
            featured_data,
            is_training=True,
            method='standard',
            feature_group='lstm'
        )
        
        return featured_data, normalized_data
    
    def temporal_train_test_split(self, data: pd.DataFrame, 
                               train_ratio: float = 0.7, 
                               val_ratio: float = 0.15) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Divise les données en ensembles d'entraînement, validation et test de manière temporelle
        
        Args:
            data: DataFrame avec les données
            train_ratio: Proportion des données pour l'entraînement
            val_ratio: Proportion des données pour la validation
            
        Returns:
            Tuple avec les DataFrames (train, val, test)
        """
        # Vérifier que les ratios sont valides
        if train_ratio + val_ratio >= 1.0:
            val_ratio = (1.0 - train_ratio) / 2
            logger.warning(f"Ratio de validation ajusté à {val_ratio}")
        
        test_ratio = 1.0 - train_ratio - val_ratio
        
        # Calculer les indices de séparation
        data_size = len(data)
        train_size = int(data_size * train_ratio)
        val_size = int(data_size * val_ratio)
        
        # Diviser en respectant l'ordre temporel
        train_data = data.iloc[:train_size]
        val_data = data.iloc[train_size:train_size+val_size]
        test_data = data.iloc[train_size+val_size:]
        
        return train_data, val_data, test_data
    
    def create_temporal_cv_folds(self, data: pd.DataFrame, 
                              n_splits: int = 5, 
                              initial_train_ratio: float = 0.5,
                              stride: Optional[int] = None) -> List[Tuple[pd.DataFrame, pd.DataFrame]]:
        """
        Crée des plis de validation croisée temporelle
        
        Args:
            data: DataFrame avec les données
            n_splits: Nombre de plis à créer
            initial_train_ratio: Ratio initial des données pour l'entraînement
            stride: Pas entre les plis (en nombre de lignes), si None, calculé automatiquement
            
        Returns:
            Liste de tuples (train_data, val_data)
        """
        data_size = len(data)
        initial_train_size = int(data_size * initial_train_ratio)
        
        # Si stride n'est pas spécifié, calculer automatiquement
        if stride is None:
            remaining_size = data_size - initial_train_size
            stride = remaining_size // n_splits
        
        cv_folds = []
        
        for i in range(n_splits):
            # Calculer les indices de séparation pour ce pli
            train_end = initial_train_size + i * stride
            val_start = train_end
            val_end = min(val_start + stride, data_size)
            
            # Créer les ensembles d'entraînement et de validation
            train_data = data.iloc[:train_end]
            val_data = data.iloc[val_start:val_end]
            
            cv_folds.append((train_data, val_data))
        
        return cv_folds
    
    def train_with_cv(self, data: pd.DataFrame, 
                    n_splits: int = 5, 
                    epochs: int = 50, 
                    batch_size: int = 32,
                    initial_train_ratio: float = 0.5,
                    patience: int = 10) -> Dict:
        """
        Entraîne le modèle avec validation croisée temporelle
        
        Args:
            data: DataFrame avec les données OHLCV brutes
            n_splits: Nombre de plis de validation croisée
            epochs: Nombre d'époques d'entraînement
            batch_size: Taille du batch
            initial_train_ratio: Ratio initial des données pour l'entraînement
            patience: Patience pour l'early stopping
            
        Returns:
            Résultats de l'entraînement
        """
        # Préparer les données
        _, normalized_data = self.prepare_data(data)
        
        # Créer les plis de validation croisée
        cv_folds = self.create_temporal_cv_folds(
            normalized_data, 
            n_splits=n_splits,
            initial_train_ratio=initial_train_ratio
        )
        
        # Résultats pour chaque pli
        cv_results = []
        
        for fold_idx, (train_data, val_data) in enumerate(cv_folds):
            logger.info(f"Entraînement sur le pli {fold_idx+1}/{n_splits}")
            
            # Préparer les données pour ce pli
            X_train, y_train = self.feature_engineering.create_multi_horizon_data(
                train_data, 
                sequence_length=self.model_params["input_length"],
                horizons=self.model_params["prediction_horizons"],
                is_training=True
            )
            
            X_val, y_val = self.feature_engineering.create_multi_horizon_data(
                val_data, 
                sequence_length=self.model_params["input_length"],
                horizons=self.model_params["prediction_horizons"],
                is_training=True
            )
            
            # Réinitialiser le modèle pour ce pli
            self.model = LSTMModel(**self.model_params)
            
            # Callbacks pour l'entraînement
            callbacks = [
                EarlyStopping(
                    monitor='val_loss',
                    patience=patience,
                    restore_best_weights=True
                ),
                ReduceLROnPlateau(
                    monitor='val_loss',
                    factor=0.5,
                    patience=patience//2,
                    min_lr=1e-6
                ),
                EarlyStoppingOnMemoryLeak(memory_threshold_mb=2000),
                ConceptDriftDetector(
                    validation_data=(X_val, y_val),
                    threshold=0.15,
                    patience=3
                )
            ]
            
            # Entraîner le modèle sur ce pli
            history = self.model.model.fit(
                x=X_train,
                y=y_train,
                epochs=epochs,
                batch_size=batch_size,
                validation_data=(X_val, y_val),
                callbacks=callbacks,
                verbose=1
            )
            
            # Évaluer le modèle sur les données de validation
            evaluation = self.model.model.evaluate(X_val, y_val, verbose=1)
            
            # Stocker les résultats de ce pli
            fold_result = {
                "fold": fold_idx + 1,
                "train_samples": len(X_train),
                "val_samples": len(X_val),
                "history": history.history,
                "val_loss": evaluation[0],
                "metrics": {f"metric_{i}": metric for i, metric in enumerate(evaluation[1:])}
            }
            
            cv_results.append(fold_result)
            
            # Sauvegarder le modèle pour ce pli
            self.model.save(os.path.join(self.output_dir, f"lstm_fold_{fold_idx+1}.h5"))
        
        # Calculer les métriques moyennes sur tous les plis
        avg_val_loss = np.mean([result["val_loss"] for result in cv_results])
        
        # Sauvegarder les résultats
        cv_summary = {
            "n_splits": n_splits,
            "avg_val_loss": float(avg_val_loss),
            "results_per_fold": cv_results,
            "timestamp": datetime.now().isoformat()
        }
        
        # Sauvegarder l'historique d'entraînement
        history_filename = f"cv_history_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        history_path = os.path.join(self.train_history_dir, history_filename)
        
        with open(history_path, 'w') as f:
            json.dump(cv_summary, f, indent=2, default=str)
        
        logger.info(f"Résultats de la validation croisée sauvegardés: {history_path}")
        
        # Entraîner le modèle final sur toutes les données sauf le dernier pli (pour test)
        self.train_final_model(normalized_data, epochs, batch_size, test_ratio=0.15)
        
        return cv_summary
    
    def train_final_model(self, data: pd.DataFrame, 
                        epochs: int = 100, 
                        batch_size: int = 32,
                        test_ratio: float = 0.15) -> Dict:
        """
        Entraîne le modèle final sur toutes les données sauf un ensemble de test
        
        Args:
            data: DataFrame avec les données normalisées
            epochs: Nombre d'époques d'entraînement
            batch_size: Taille du batch
            test_ratio: Ratio des données pour le test final
            
        Returns:
            Résultats de l'entraînement
        """
        # Diviser les données en train+val et test
        train_size = int(len(data) * (1 - test_ratio))
        train_val_data = data.iloc[:train_size]
        test_data = data.iloc[train_size:]
        
        # Diviser train_val en train et validation
        train_data, val_data, _ = self.temporal_train_test_split(
            train_val_data, 
            train_ratio=0.8,
            val_ratio=0.2
        )
        
        # Préparer les données
        X_train, y_train = self.feature_engineering.create_multi_horizon_data(
            train_data, 
            sequence_length=self.model_params["input_length"],
            horizons=self.model_params["prediction_horizons"],
            is_training=True
        )
        
        X_val, y_val = self.feature_engineering.create_multi_horizon_data(
            val_data, 
            sequence_length=self.model_params["input_length"],
            horizons=self.model_params["prediction_horizons"],
            is_training=True
        )
        
        X_test, y_test = self.feature_engineering.create_multi_horizon_data(
            test_data, 
            sequence_length=self.model_params["input_length"],
            horizons=self.model_params["prediction_horizons"],
            is_training=True
        )
        
        # Update feature_dim based on normalized data shape
        self.model_params['feature_dim'] = X_train.shape[2]
        
        # Réinitialiser le modèle
        self.model = LSTMModel(**self.model_params)
        
        # Callbacks pour l'entraînement
        callbacks = [
            EarlyStopping(
                monitor='val_loss',
                patience=20,
                restore_best_weights=True
            ),
            ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=10,
                min_lr=1e-6
            ),
            ModelCheckpoint(
                filepath=os.path.join(self.output_dir, "production", "lstm_best.keras"),  # changed extension to .keras
                monitor='val_loss',
                save_best_only=True
            ),
            EarlyStoppingOnMemoryLeak(memory_threshold_mb=2000)
        ]
        
        # Entraîner le modèle final
        history = self.model.model.fit(
            x=X_train,
            y=y_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_data=(X_val, y_val),
            callbacks=callbacks,
            verbose=1
        )
        
        # Évaluer le modèle sur les données de test
        test_evaluation = self.model.model.evaluate(X_test, y_test, verbose=1)
        
        # Calculer des métriques supplémentaires pour le test
        y_pred = self.model.model.predict(X_test)
        
        # Pour chaque horizon, calculer la précision de la direction
        direction_accuracies = []
        
        for h_idx, horizon in enumerate(self.model_params["prediction_horizons"]):
            # Indice de base pour la direction dans les sorties
            base_idx = h_idx * 4
            
            # Prédictions de direction (binaires)
            y_true_direction = y_test[base_idx]
            y_pred_direction = (y_pred[base_idx] > 0.5).astype(int)
            
            # Calculer l'accuracy
            accuracy = np.mean(y_true_direction == y_pred_direction.flatten())
            direction_accuracies.append(accuracy)
        
        # Sauvegarder les résultats
        final_results = {
            "train_samples": len(X_train),
            "val_samples": len(X_val),
            "test_samples": len(X_test),
            "history": history.history,
            "test_loss": test_evaluation,
            "direction_accuracies": direction_accuracies,
            "timestamp": datetime.now().isoformat()
        }
        
        # Sauvegarder l'historique d'entraînement
        history_filename = f"final_model_history_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        history_path = os.path.join(self.train_history_dir, history_filename)
        
        with open(history_path, 'w') as f:
            json.dump(final_results, f, indent=2, default=str)
        
        logger.info(f"Résultats du modèle final sauvegardés: {history_path}")
        
        # Sauvegarder le modèle final
        self.model.save(os.path.join(self.output_dir, "production", "lstm_final.h5"))
        
        # Générer des visualisations
        self._generate_training_visualizations(history.history, final_results)
        
        return final_results
    
    def _generate_training_visualizations(self, history: Dict, results: Dict) -> None:
        """
        Génère des visualisations de l'entraînement du modèle
        
        Args:
            history: Historique d'entraînement
            results: Résultats du modèle
        """
        # Créer le répertoire pour les visualisations
        viz_dir = os.path.join(self.output_dir, "visualizations")
        os.makedirs(viz_dir, exist_ok=True)
        
        # 1. Courbe d'apprentissage (loss)
        plt.figure(figsize=(12, 6))
        plt.plot(history['loss'], label='Train Loss')
        plt.plot(history['val_loss'], label='Validation Loss')
        plt.title('Courbe d\'apprentissage du modèle LSTM')
        plt.xlabel('Époque')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Ajouter des annotations
        min_val_loss = min(history['val_loss'])
        min_val_loss_epoch = history['val_loss'].index(min_val_loss)
        
        plt.annotate(f'Min Val Loss: {min_val_loss:.4f}',
                    xy=(min_val_loss_epoch, min_val_loss),
                    xytext=(min_val_loss_epoch+5, min_val_loss*1.1),
                    arrowprops=dict(facecolor='black', shrink=0.05, width=1.5, headwidth=8),
                    fontsize=10)
        
        # Sauvegarder la figure
        plt.savefig(os.path.join(viz_dir, "learning_curve.png"))
        plt.close()
        
        # 2. Précision de la direction par horizon
        plt.figure(figsize=(10, 6))
        horizons = self.model_params["prediction_horizons"]
        accuracies = results["direction_accuracies"]
        
        plt.bar(range(len(horizons)), accuracies, color='skyblue')
        plt.xticks(range(len(horizons)), [f"{h}" for h in horizons])
        plt.title('Précision de la Direction par Horizon')
        plt.xlabel('Horizon (périodes)')
        plt.ylabel('Précision')
        plt.ylim([0, 1])
        
        # Ajouter les valeurs sur les barres
        for i, v in enumerate(accuracies):
            plt.text(i, v + 0.02, f"{v:.2f}", ha='center')
        
        plt.grid(True, alpha=0.3, axis='y')
        plt.tight_layout()
        
        # Sauvegarder la figure
        plt.savefig(os.path.join(viz_dir, "direction_accuracy.png"))
        plt.close()
        
        # 3. Évolution du taux d'apprentissage
        if 'lr' in history:
            plt.figure(figsize=(10, 4))
            plt.semilogy(history['lr'])
            plt.title('Évolution du Taux d\'Apprentissage')
            plt.xlabel('Époque')
            plt.ylabel('Learning Rate')
            plt.grid(True, alpha=0.3)
            
            # Sauvegarder la figure
            plt.savefig(os.path.join(viz_dir, "learning_rate.png"))
            plt.close()