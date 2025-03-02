# ai/models/continuous_learning.py
"""
Module d'apprentissage continu pour le modèle LSTM
Permet au modèle de s'adapter aux nouvelles données du marché
sans oublier les connaissances précédentes (éviter l'oubli catastrophique)
"""
import os
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Union, Optional
from datetime import datetime, timedelta
import json
import tensorflow as tf
import pickle
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
import random
from collections import deque

from ai.models.lstm_model import LSTMModel
from ai.models.feature_engineering import FeatureEngineering
from ai.models.model_validator import ModelValidator
from config.config import DATA_DIR
from utils.logger import setup_logger

logger = setup_logger("continuous_learning")

class ExperienceBuffer:
    """
    Tampon d'expérience pour stocker les exemples d'entraînement passés
    Utilisé pour éviter l'oubli catastrophique en mélangeant les anciennes 
    et nouvelles données lors de l'apprentissage
    """
    def __init__(self, max_size: int = 10000, importance_sampling: bool = True):
        """
        Initialise le tampon d'expérience
        
        Args:
            max_size: Taille maximale du tampon
            importance_sampling: Utiliser l'échantillonnage par importance
        """
        self.buffer = deque(maxlen=max_size)
        self.importance_sampling = importance_sampling
        self.importance_scores = deque(maxlen=max_size)
        
    def add(self, example: Tuple, importance: float = 1.0):
        """
        Ajoute un exemple au tampon
        
        Args:
            example: Tuple (X, y) d'un exemple d'entraînement
            importance: Score d'importance de l'exemple (défaut: 1.0)
        """
        self.buffer.append(example)
        self.importance_scores.append(importance)
    
    def sample(self, batch_size: int) -> List[Tuple]:
        """
        Échantillonne des exemples du tampon
        
        Args:
            batch_size: Nombre d'exemples à échantillonner
            
        Returns:
            Liste d'exemples échantillonnés
        """
        if len(self.buffer) == 0:
            return []
        
        # Limiter la taille du batch à la taille du tampon
        batch_size = min(batch_size, len(self.buffer))
        
        if self.importance_sampling:
            # Normaliser les scores d'importance
            total_importance = sum(self.importance_scores)
            if total_importance > 0:
                probs = [score / total_importance for score in self.importance_scores]
                indices = np.random.choice(len(self.buffer), batch_size, replace=False, p=probs)
                return [self.buffer[i] for i in indices]
        
        # Échantillonnage uniforme si pas d'échantillonnage par importance
        return random.sample(list(self.buffer), batch_size)
    
    def clear(self):
        """Vide le tampon"""
        self.buffer.clear()
        self.importance_scores.clear()
    
    def __len__(self):
        return len(self.buffer)
    
    def save(self, filepath: str):
        """
        Sauvegarde le tampon d'expérience sur disque
        
        Args:
            filepath: Chemin du fichier de sauvegarde
        """
        try:
            with open(filepath, 'wb') as f:
                pickle.dump({
                    'buffer': list(self.buffer),
                    'importance_scores': list(self.importance_scores)
                }, f)
            logger.info(f"Tampon d'expérience sauvegardé: {filepath}")
        except Exception as e:
            logger.error(f"Erreur lors de la sauvegarde du tampon: {str(e)}")
    
    def load(self, filepath: str):
        """
        Charge le tampon d'expérience depuis le disque
        
        Args:
            filepath: Chemin du fichier de sauvegarde
        """
        if not os.path.exists(filepath):
            logger.warning(f"Fichier du tampon non trouvé: {filepath}")
            return
        
        try:
            with open(filepath, 'rb') as f:
                data = pickle.load(f)
                self.buffer = deque(data['buffer'], maxlen=self.buffer.maxlen)
                self.importance_scores = deque(data['importance_scores'], maxlen=self.importance_scores.maxlen)
            logger.info(f"Tampon d'expérience chargé: {filepath} ({len(self.buffer)} exemples)")
        except Exception as e:
            logger.error(f"Erreur lors du chargement du tampon: {str(e)}")


class ConceptDriftDetector:
    """
    Détecteur de dérive conceptuelle pour surveiller les changements dans les données
    et déterminer quand le modèle doit être mis à jour
    """
    def __init__(self, window_size: int = 50, threshold: float = 0.15):
        """
        Initialise le détecteur de dérive conceptuelle
        
        Args:
            window_size: Taille de la fenêtre d'observation
            threshold: Seuil de dérive (écart relatif)
        """
        self.window_size = window_size
        self.threshold = threshold
        self.reference_distribution = None
        self.current_window = []
        
    def add_observation(self, observation: Dict) -> bool:
        """
        Ajoute une observation et vérifie s'il y a une dérive
        
        Args:
            observation: Dictionnaire avec les métriques à surveiller
            
        Returns:
            True s'il y a une dérive conceptuelle, False sinon
        """
        # Ajouter l'observation à la fenêtre courante
        self.current_window.append(observation)
        
        # Garder seulement les 'window_size' dernières observations
        if len(self.current_window) > self.window_size:
            self.current_window.pop(0)
        
        # Si la fenêtre n'est pas complète ou pas de référence, pas de dérive
        if len(self.current_window) < self.window_size or self.reference_distribution is None:
            return False
        
        # Calculer la distribution actuelle
        current_distribution = self._calculate_distribution(self.current_window)
        
        # Calculer la divergence entre les distributions
        drift_score = self._calculate_drift(self.reference_distribution, current_distribution)
        
        return drift_score > self.threshold
    
    def update_reference(self, observations: List[Dict] = None):
        """
        Met à jour la distribution de référence
        
        Args:
            observations: Liste d'observations pour la référence (utilise la fenêtre courante si None)
        """
        if observations is None:
            if len(self.current_window) < self.window_size:
                logger.warning("Fenêtre courante trop petite pour servir de référence")
                return
            observations = self.current_window
        
        self.reference_distribution = self._calculate_distribution(observations)
        logger.info("Distribution de référence mise à jour")
    
    def _calculate_distribution(self, observations: List[Dict]) -> Dict:
        """
        Calcule la distribution des métriques dans les observations
        
        Args:
            observations: Liste d'observations
            
        Returns:
            Dictionnaire avec les statistiques de la distribution
        """
        # Extraire toutes les métriques disponibles
        metrics = {}
        
        # Pour chaque observation, collecter les métriques
        for obs in observations:
            for key, value in obs.items():
                if isinstance(value, (int, float)):
                    if key not in metrics:
                        metrics[key] = []
                    metrics[key].append(value)
        
        # Calculer les statistiques pour chaque métrique
        distribution = {}
        for key, values in metrics.items():
            distribution[key] = {
                'mean': np.mean(values),
                'std': np.std(values),
                'min': np.min(values),
                'max': np.max(values),
                'median': np.median(values)
            }
        
        return distribution
    
    def _calculate_drift(self, reference: Dict, current: Dict) -> float:
        """
        Calcule le score de dérive entre deux distributions
        
        Args:
            reference: Distribution de référence
            current: Distribution courante
            
        Returns:
            Score de dérive (0-1)
        """
        # Calculer la divergence pour chaque métrique commune
        metric_drifts = []
        
        for key in reference.keys():
            if key in current:
                ref_metrics = reference[key]
                curr_metrics = current[key]
                
                # Calculer la divergence relative sur la moyenne et l'écart-type
                mean_diff = abs(ref_metrics['mean'] - curr_metrics['mean'])
                mean_drift = mean_diff / (abs(ref_metrics['mean']) + 1e-10)
                
                std_diff = abs(ref_metrics['std'] - curr_metrics['std'])
                std_drift = std_diff / (abs(ref_metrics['std']) + 1e-10)
                
                # Combiner les divergences
                metric_drift = (mean_drift + std_drift) / 2
                metric_drifts.append(metric_drift)
        
        # Retourner la moyenne des divergences
        if not metric_drifts:
            return 0
        
        return np.mean(metric_drifts)


class ContinuousLearning:
    """
    Gère l'apprentissage continu du modèle LSTM
    """
    def __init__(self, model: Optional[LSTMModel] = None,
                feature_engineering: Optional[FeatureEngineering] = None,
                experience_buffer_size: int = 10000,
                drift_threshold: float = 0.15,
                drift_window_size: int = 50):
        """
        Initialise le module d'apprentissage continu
        
        Args:
            model: Instance du modèle LSTM à adapter
            feature_engineering: Instance du module d'ingénierie des caractéristiques
            experience_buffer_size: Taille du tampon d'expérience
            drift_threshold: Seuil de détection de dérive
            drift_window_size: Taille de la fenêtre pour la détection de dérive
        """
        # Composants principaux
        self.model = model
        self.feature_engineering = feature_engineering or FeatureEngineering()
        
        # Tampon d'expérience
        self.experience_buffer = ExperienceBuffer(
            max_size=experience_buffer_size,
            importance_sampling=True
        )
        
        # Détecteur de dérive conceptuelle
        self.drift_detector = ConceptDriftDetector(
            window_size=drift_window_size,
            threshold=drift_threshold
        )
        
        # Répertoires pour les modèles et données
        self.models_dir = os.path.join(DATA_DIR, "models")
        self.buffer_dir = os.path.join(self.models_dir, "experience_buffer")
        
        # Créer les répertoires si nécessaires
        os.makedirs(self.models_dir, exist_ok=True)
        os.makedirs(self.buffer_dir, exist_ok=True)
        
        # Historique des mises à jour
        self.update_history = []
        
        # Chargement du tampon d'expérience s'il existe
        buffer_path = os.path.join(self.buffer_dir, "experience_buffer.pkl")
        if os.path.exists(buffer_path):
            self.experience_buffer.load(buffer_path)
    
    def process_new_data(self, data: pd.DataFrame, min_samples: int = 100,
                       max_drift_score: float = 0.3) -> Dict:
        """
        Traite de nouvelles données et met à jour le modèle si nécessaire
        
        Args:
            data: DataFrame avec les nouvelles données OHLCV
            min_samples: Nombre minimum d'échantillons pour la mise à jour
            max_drift_score: Score de dérive maximum toléré
            
        Returns:
            Résultats du traitement
        """
        # Si le modèle n'est pas initialisé, charger le modèle de production
        if self.model is None:
            model_path = os.path.join(self.models_dir, "production", "lstm_final.h5")
            if os.path.exists(model_path):
                self.model = LSTMModel()
                self.model.load(model_path)
                logger.info(f"Modèle chargé: {model_path}")
            else:
                logger.error("Aucun modèle disponible")
                return {"success": False, "message": "Aucun modèle disponible"}
        
        # 1. Préparer les données
        try:
            featured_data, normalized_data = self._prepare_data(data)
        except Exception as e:
            logger.error(f"Erreur lors de la préparation des données: {str(e)}")
            return {"success": False, "message": f"Erreur de préparation: {str(e)}"}
        
        # 2. Évaluer la performance du modèle sur les nouvelles données
        validator = ModelValidator(self.model, self.feature_engineering)
        evaluation = validator.evaluate_on_test_set(normalized_data)
        
        # 3. Vérifier la dérive conceptuelle
        drift_observation = {
            'loss': evaluation['loss'],
            'timestamp': datetime.now().timestamp()
        }
        
        # Pour chaque horizon, ajouter les métriques de direction
        for horizon_key, horizon_metrics in evaluation['horizons'].items():
            drift_observation[f"{horizon_key}_direction_accuracy"] = horizon_metrics['direction']['accuracy']
        
        # Ajouter l'observation au détecteur de dérive
        drift_detected = self.drift_detector.add_observation(drift_observation)
        
        # 4. Décider si une mise à jour du modèle est nécessaire
        update_needed = drift_detected
        
        # Même si pas de dérive, mettre à jour si la performance est mauvaise
        if not update_needed:
            poor_performance = False
            for horizon_key, metrics in evaluation['horizons'].items():
                direction_accuracy = metrics['direction']['accuracy']
                if direction_accuracy < 0.52:  # Seuil légèrement au-dessus de 50% (hasard)
                    poor_performance = True
                    break
            
            if poor_performance:
                logger.info("Mise à jour nécessaire due à une faible performance")
                update_needed = True
        
        # 5. Ajouter les données au tampon d'expérience
        self._add_to_experience_buffer(normalized_data)
        
        # 6. Mettre à jour le modèle si nécessaire
        if update_needed and len(self.experience_buffer) >= min_samples:
            update_result = self._update_model(
                new_data=normalized_data,
                drift_score=drift_observation.get('loss', 1.0)
            )
            
            # Réinitialiser le détecteur de dérive après la mise à jour
            self.drift_detector.update_reference()
            
            # Sauvegarder le tampon d'expérience
            buffer_path = os.path.join(self.buffer_dir, "experience_buffer.pkl")
            self.experience_buffer.save(buffer_path)
            
            # Ajouter à l'historique des mises à jour
            update_info = {
                "timestamp": datetime.now().isoformat(),
                "drift_detected": drift_detected,
                "evaluation_before": evaluation,
                "update_result": update_result
            }
            self.update_history.append(update_info)
            
            # Sauvegarder l'historique des mises à jour
            self._save_update_history()
            
            return {
                "success": True,
                "updated": True,
                "drift_detected": drift_detected,
                "evaluation": evaluation,
                "update_result": update_result
            }
        
        return {
            "success": True,
            "updated": False,
            "drift_detected": drift_detected,
            "evaluation": evaluation
        }
    
    def _prepare_data(self, data: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Prépare les données pour le traitement
        
        Args:
            data: DataFrame avec les données OHLCV brutes
            
        Returns:
            Tuple (données avec caractéristiques, données normalisées)
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
            is_training=False,  # Utiliser les scalers existants
            method='standard',
            feature_group='lstm'
        )
        
        return featured_data, normalized_data
    
    def _add_to_experience_buffer(self, data: pd.DataFrame) -> None:
        """
        Ajoute les données au tampon d'expérience
        
        Args:
            data: DataFrame avec les données normalisées
        """
        # Créer des exemples d'entraînement
        X, y = self.feature_engineering.create_multi_horizon_data(
            data,
            sequence_length=self.model.input_length,
            horizons=self.model.prediction_horizons,
            is_training=True
        )
        
        # Si la création d'exemples a échoué
        if len(X) == 0:
            logger.warning("Aucun exemple créé pour le tampon d'expérience")
            return
        
        # Évaluer l'importance de chaque exemple
        importances = self._calculate_example_importance(X, y)
        
        # Ajouter les exemples au tampon
        for i in range(len(X)):
            example = (X[i:i+1], [y_arr[i:i+1] for y_arr in y])
            importance = importances[i] if i < len(importances) else 1.0
            self.experience_buffer.add(example, importance)
        
        logger.info(f"Ajouté {len(X)} exemples au tampon d'expérience (taille: {len(self.experience_buffer)})")
    
    def _calculate_example_importance(self, X: np.ndarray, y: List[np.ndarray]) -> List[float]:
        """
        Calcule l'importance de chaque exemple pour l'échantillonnage par importance
        
        Args:
            X: Données d'entrée
            y: Liste des sorties cibles
            
        Returns:
            Liste des scores d'importance
        """
        # Si le modèle n'est pas disponible, importance uniforme
        if self.model is None:
            return [1.0] * len(X)
        
        # Faire des prédictions avec le modèle actuel
        predictions = self.model.model.predict(X)
        
        # Calculer l'erreur pour chaque exemple
        importances = []
        
        # Pour chaque exemple
        for i in range(len(X)):
            # Calculer l'erreur sur les sorties de direction (les plus importantes)
            error_sum = 0
            for h_idx in range(len(self.model.prediction_horizons)):
                # Indice de base pour la direction de cet horizon
                base_idx = h_idx * 4
                
                # Erreur de direction (classification binaire)
                y_true = y[base_idx][i]
                y_pred = predictions[base_idx][i]
                
                # Erreur quadratique
                error = (y_true - y_pred) ** 2
                error_sum += float(error)
            
            # L'importance est proportionnelle à l'erreur
            # Les exemples difficiles (erreur élevée) ont plus d'importance
            importance = 1.0 + error_sum  # Ajouter 1 pour éviter les importances nulles
            importances.append(importance)
        
        return importances
    
    def _update_model(self, new_data: pd.DataFrame, drift_score: float) -> Dict:
        """
        Met à jour le modèle avec de nouvelles données et des exemples du tampon
        
        Args:
            new_data: DataFrame avec les nouvelles données normalisées
            drift_score: Score de dérive des nouvelles données
            
        Returns:
            Résultats de la mise à jour
        """
        # Sauvegarde des poids actuels
        original_weights = self.model.model.get_weights()
        
        # 1. Préparer les données pour la mise à jour
        # Nouvelles données
        X_new, y_new = self.feature_engineering.create_multi_horizon_data(
            new_data,
            sequence_length=self.model.input_length,
            horizons=self.model.prediction_horizons,
            is_training=True
        )
        
        # 2. Échantillonner des exemples du tampon d'expérience
        # Plus la dérive est importante, plus on utilise d'exemples anciens
        buffer_ratio = min(0.8, drift_score)  # Max 80% d'exemples anciens
        buffer_samples = int(len(X_new) * buffer_ratio / (1 - buffer_ratio))
        
        buffered_examples = self.experience_buffer.sample(buffer_samples)
        
        if not buffered_examples:
            logger.warning("Tampon d'expérience vide, mise à jour uniquement avec les nouvelles données")
            
            # Mettre à jour avec les nouvelles données uniquement
            return self._perform_update_step(X_new, y_new)
        
        # 3. Combiner les nouvelles données et les exemples du tampon
        X_buffer = np.vstack([example[0] for example in buffered_examples])
        y_buffer = []
        
        # Pour chaque sortie, combiner les exemples
        for output_idx in range(len(y_new)):
            y_output = np.vstack([example[1][output_idx] for example in buffered_examples])
            y_buffer.append(y_output)
        
        # Combiner avec les nouvelles données
        X_combined = np.vstack([X_new, X_buffer])
        y_combined = []
        
        for output_idx in range(len(y_new)):
            y_combined.append(np.vstack([y_new[output_idx], y_buffer[output_idx]]))
        
        # 4. Effectuer la mise à jour
        return self._perform_update_step(X_combined, y_combined, original_weights)
    
    def _perform_update_step(self, X: np.ndarray, y: List[np.ndarray], 
                          original_weights: Optional[List] = None) -> Dict:
        """
        Effectue une étape de mise à jour du modèle
        
        Args:
            X: Données d'entrée combinées
            y: Liste des sorties cibles combinées
            original_weights: Poids originaux du modèle (pour revenir en arrière si nécessaire)
            
        Returns:
            Résultats de la mise à jour
        """
        # Paramètres de mise à jour
        epochs = 5
        batch_size = 32
        learning_rate = 0.0005  # Taux d'apprentissage réduit pour la mise à jour
        
        # Réduire le taux d'apprentissage
        tf.keras.backend.set_value(self.model.model.optimizer.learning_rate, learning_rate)
        
        # Callbacks pour la mise à jour
        callbacks = [
            EarlyStopping(
                monitor='loss',
                patience=2,
                restore_best_weights=True
            ),
            ReduceLROnPlateau(
                monitor='loss',
                factor=0.5,
                patience=1,
                min_lr=1e-6
            )
        ]
        
        # Évaluer avant la mise à jour
        evaluation_before = self.model.model.evaluate(X, y, verbose=0)
        
        # Effectuer la mise à jour
        history = self.model.model.fit(
            x=X,
            y=y,
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            verbose=1
        )
        
        # Évaluer après la mise à jour
        evaluation_after = self.model.model.evaluate(X, y, verbose=0)
        
        # Vérifier si la mise à jour a amélioré le modèle
        if evaluation_after[0] > evaluation_before[0] * 1.1:  # Dégradation de plus de 10%
            logger.warning("La mise à jour a dégradé les performances, retour aux poids précédents")
            if original_weights is not None:
                self.model.model.set_weights(original_weights)
            
            return {
                "success": False,
                "message": "Dégradation des performances",
                "evaluation_before": evaluation_before[0],
                "evaluation_after": evaluation_after[0]
            }
        
        # Sauvegarder le modèle mis à jour
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        update_model_path = os.path.join(self.models_dir, "updates", f"lstm_update_{timestamp}.h5")
        
        # Créer le répertoire si nécessaire
        os.makedirs(os.path.dirname(update_model_path), exist_ok=True)
        
        # Sauvegarder le modèle
        self.model.save(update_model_path)
        
        # Également mettre à jour le modèle de production
        production_path = os.path.join(self.models_dir, "production", "lstm_final.h5")
        self.model.save(production_path)
        
        return {
            "success": True,
            "message": "Mise à jour réussie",
            "evaluation_before": evaluation_before[0],
            "evaluation_after": evaluation_after[0],
            "improvement": (evaluation_before[0] - evaluation_after[0]) / evaluation_before[0] * 100,
            "epochs_trained": len(history.history['loss']),
            "model_path": update_model_path,
            "samples_used": len(X)
        }
    
    def _save_update_history(self) -> None:
        """Sauvegarde l'historique des mises à jour"""
        history_path = os.path.join(self.models_dir, "updates", "update_history.json")
        
        # Créer le répertoire si nécessaire
        os.makedirs(os.path.dirname(history_path), exist_ok=True)
        
        try:
            with open(history_path, 'w') as f:
                json.dump(self.update_history, f, indent=2, default=str)
            logger.info(f"Historique des mises à jour sauvegardé: {history_path}")
        except Exception as e:
            logger.error(f"Erreur lors de la sauvegarde de l'historique: {str(e)}")