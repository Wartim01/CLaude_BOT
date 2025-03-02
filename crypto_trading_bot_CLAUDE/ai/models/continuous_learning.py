"""
Système d'apprentissage continu avancé avec protection contre l'oubli catastrophique
et détection de concept drift pour l'adaptation automatique aux changements de marché
"""
import os
import numpy as np
import pandas as pd
import tensorflow as tf
import pickle
import json
from typing import Dict, List, Tuple, Union, Optional, Any
from datetime import datetime, timedelta
from collections import deque
import random
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from scipy.stats import mannwhitneyu, ks_2samp
import shutil

from config.config import DATA_DIR
from utils.logger import setup_logger

logger = setup_logger("advanced_continuous_learning")

class ReplayMemory:
    """
    Mémoire de rejeu avec échantillonnage prioritaire
    Stocke les exemples d'entraînement passés pour éviter l'oubli catastrophique
    """
    def __init__(self, max_size: int = 10000, alpha: float = 0.6, beta: float = 0.4):
        """
        Initialise la mémoire de rejeu
        
        Args:
            max_size: Capacité maximale de la mémoire
            alpha: Facteur d'exposition pour le calcul des priorités (0 = échantillonnage uniforme)
            beta: Facteur de correction pour le biais de l'échantillonnage prioritaire
        """
        self.max_size = max_size
        self.memory = deque(maxlen=max_size)
        self.priorities = deque(maxlen=max_size)
        self.alpha = alpha
        self.beta = beta
        self.epsilon = 1e-6  # Petite valeur pour éviter les priorités nulles
        
        # Métadonnées pour les statistiques et le diagnostic
        self.insertion_timestamps = deque(maxlen=max_size)
        self.memory_clusters = {}  # Pour l'organisation par concept/régime
        
        # Pour suivre les distributions des caractéristiques
        self.feature_stats = {
            "means": {},
            "stds": {},
            "mins": {},
            "maxs": {}
        }
    
    def add(self, experience: Tuple, priority: float = None) -> None:
        """
        Ajoute une expérience à la mémoire
        
        Args:
            experience: Tuple (X, y) d'un exemple d'entraînement
            priority: Priorité de l'exemple (si None, utilise la priorité maximale actuelle)
        """
        if priority is None:
            priority = max(self.priorities) if self.priorities else 1.0
            
        self.memory.append(experience)
        self.priorities.append(priority)
        self.insertion_timestamps.append(datetime.now().isoformat())
        
        # Mettre à jour les statistiques de caractéristiques si l'expérience contient des données
        if len(experience) > 0 and isinstance(experience[0], np.ndarray) and experience[0].size > 0:
            self._update_feature_stats(experience[0])
    
    def sample(self, batch_size: int) -> List[Tuple]:
        """
        Échantillonne un batch d'expériences selon leur priorité
        
        Args:
            batch_size: Taille du batch à échantillonner
            
        Returns:
            Liste d'expériences échantillonnées et leurs indices
        """
        if len(self.memory) == 0:
            return []
        
        batch_size = min(batch_size, len(self.memory))
        
        # Calculer les probabilités d'échantillonnage selon les priorités
        priorities = np.array(self.priorities)
        probabilities = priorities ** self.alpha
        probabilities /= np.sum(probabilities)
        
        # Échantillonner les indices selon les probabilités
        indices = np.random.choice(len(self.memory), batch_size, replace=False, p=probabilities)
        
        # Calculer les poids d'importance pour la correction du biais
        weights = (len(self.memory) * probabilities[indices]) ** (-self.beta)
        weights /= np.max(weights)  # Normaliser à 1
        
        # Récupérer les expériences échantillonnées
        batch = [self.memory[i] for i in indices]
        
        return batch, indices, weights
    
    def update_priorities(self, indices: List[int], errors: List[float]) -> None:
        """
        Met à jour les priorités des expériences en fonction des erreurs d'entraînement
        
        Args:
            indices: Indices des expériences à mettre à jour
            errors: Erreurs d'entraînement correspondantes
        """
        for i, idx in enumerate(indices):
            if idx < len(self.priorities):
                # Priorité = erreur + epsilon (pour éviter les priorités nulles)
                self.priorities[idx] = errors[i] + self.epsilon
    
    def organize_by_clusters(self, n_clusters: int = 5) -> Dict:
        """
        Organise la mémoire en clusters pour identifier différents régimes de marché
        
        Args:
            n_clusters: Nombre de clusters à former
            
        Returns:
            Dictionnaire des clusters
        """
        if len(self.memory) < n_clusters * 10:
            return {}  # Pas assez de données pour le clustering
        
        # Extraire les caractéristiques des expériences
        features = []
        for exp in self.memory:
            if len(exp) > 0 and isinstance(exp[0], np.ndarray):
                # Prendre la moyenne des caractéristiques temporelles
                features.append(np.mean(exp[0], axis=0).flatten())
        
        if not features:
            return {}
        
        features = np.array(features)
        
        # Déterminer le nombre optimal de clusters si non spécifié
        if n_clusters is None:
            n_clusters = self._find_optimal_clusters(features, max_clusters=10)
        
        # Appliquer KMeans
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        clusters = kmeans.fit_predict(features)
        
        # Organiser la mémoire par cluster
        self.memory_clusters = {i: [] for i in range(n_clusters)}
        
        for i, cluster_id in enumerate(clusters):
            if i < len(self.memory):
                self.memory_clusters[cluster_id].append(i)
        
        # Calculer les statistiques des clusters
        cluster_stats = {}
        for cluster_id, indices in self.memory_clusters.items():
            if indices:
                priorities = [self.priorities[i] for i in indices]
                timestamps = [self.insertion_timestamps[i] for i in indices]
                
                cluster_stats[cluster_id] = {
                    "size": len(indices),
                    "avg_priority": np.mean(priorities),
                    "newest": min(timestamps),
                    "oldest": max(timestamps)
                }
        
        return cluster_stats
    
    def get_balanced_batch(self, batch_size: int, recency_weight: float = 0.3) -> List[Tuple]:
        """
        Échantillonne un batch équilibré qui combine expériences récentes et anciennes
        
        Args:
            batch_size: Taille du batch à échantillonner
            recency_weight: Poids pour les expériences récentes vs. diverses
            
        Returns:
            Liste d'expériences échantillonnées
        """
        if len(self.memory) == 0:
            return []
        
        batch_size = min(batch_size, len(self.memory))
        
        # Nombre d'expériences récentes et diverses
        recent_count = int(batch_size * recency_weight)
        diverse_count = batch_size - recent_count
        
        # Échantillonner les expériences récentes
        recent_indices = np.argsort([i for i in range(len(self.memory))])[-recent_count:]
        recent_batch = [self.memory[i] for i in recent_indices]
        
        # Échantillonner des expériences diverses selon les priorités
        diverse_batch = []
        if diverse_count > 0 and len(self.memory) > recent_count:
            # Exclure les échantillons récents déjà sélectionnés
            remaining_indices = [i for i in range(len(self.memory)) if i not in recent_indices]
            remaining_priorities = [self.priorities[i] for i in remaining_indices]
            
            # Calculer les probabilités
            probabilities = np.array(remaining_priorities) ** self.alpha
            probabilities /= np.sum(probabilities)
            
            # Échantillonner
            selected_indices = np.random.choice(
                remaining_indices, 
                min(diverse_count, len(remaining_indices)), 
                replace=False, 
                p=probabilities
            )
            
            diverse_batch = [self.memory[i] for i in selected_indices]
        
        # Combiner et mélanger
        combined_batch = recent_batch + diverse_batch
        random.shuffle(combined_batch)
        
        return combined_batch
    
    def _update_feature_stats(self, X: np.ndarray) -> None:
        """
        Met à jour les statistiques des caractéristiques
        
        Args:
            X: Entrée du modèle
        """
        # Si X est 3D (batch, sequence, features), réduire à 2D
        if X.ndim == 3:
            X_flat = X.reshape(-1, X.shape[-1])
        else:
            X_flat = X
        
        # Mettre à jour les statistiques
        for i in range(X_flat.shape[1]):
            feature_values = X_flat[:, i]
            
            if i not in self.feature_stats["means"]:
                self.feature_stats["means"][i] = []
                self.feature_stats["stds"][i] = []
                self.feature_stats["mins"][i] = []
                self.feature_stats["maxs"][i] = []
            
            self.feature_stats["means"][i].append(np.mean(feature_values))
            self.feature_stats["stds"][i].append(np.std(feature_values))
            self.feature_stats["mins"][i].append(np.min(feature_values))
            self.feature_stats["maxs"][i].append(np.max(feature_values))
            
            # Garder seulement les 100 dernières valeurs
            for key in ["means", "stds", "mins", "maxs"]:
                self.feature_stats[key][i] = self.feature_stats[key][i][-100:]
    
    def _find_optimal_clusters(self, features: np.ndarray, max_clusters: int = 10) -> int:
        """
        Trouve le nombre optimal de clusters avec la méthode du score de silhouette
        
        Args:
            features: Caractéristiques à clusteriser
            max_clusters: Nombre maximum de clusters à tester
            
        Returns:
            Nombre optimal de clusters
        """
        if len(features) < max_clusters * 2:
            return max(2, len(features) // 5)
        
        silhouette_scores = []
        cluster_range = range(2, min(max_clusters, len(features) // 10) + 1)
        
        for n_clusters in cluster_range:
            try:
                kmeans = KMeans(n_clusters=n_clusters, random_state=42)
                cluster_labels = kmeans.fit_predict(features)
                silhouette_avg = silhouette_score(features, cluster_labels)
                silhouette_scores.append(silhouette_avg)
            except:
                silhouette_scores.append(-1)
        
        if not silhouette_scores or max(silhouette_scores) < 0:
            return 3  # Valeur par défaut
        
        return cluster_range[np.argmax(silhouette_scores)]
    
    def save(self, filepath: str) -> None:
        """
        Sauvegarde la mémoire de rejeu sur disque
        
        Args:
            filepath: Chemin du fichier de sauvegarde
        """
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        # Préparer les données à sauvegarder
        save_data = {
            "memory": list(self.memory),
            "priorities": list(self.priorities),
            "insertion_timestamps": list(self.insertion_timestamps),
            "feature_stats": self.feature_stats,
            "alpha": self.alpha,
            "beta": self.beta,
            "max_size": self.max_size
        }
        
        try:
            with open(filepath, 'wb') as f:
                pickle.dump(save_data, f)
            logger.info(f"Mémoire de rejeu sauvegardée: {filepath}")
        except Exception as e:
            logger.error(f"Erreur lors de la sauvegarde de la mémoire de rejeu: {str(e)}")
    
    def load(self, filepath: str) -> bool:
        """
        Charge la mémoire de rejeu depuis le disque
        
        Args:
            filepath: Chemin du fichier de sauvegarde
            
        Returns:
            Succès du chargement
        """
        if not os.path.exists(filepath):
            logger.warning(f"Fichier de mémoire de rejeu non trouvé: {filepath}")
            return False
        
        try:
            with open(filepath, 'rb') as f:
                save_data = pickle.load(f)
            
            self.memory = deque(save_data["memory"], maxlen=save_data["max_size"])
            self.priorities = deque(save_data["priorities"], maxlen=save_data["max_size"])
            self.insertion_timestamps = deque(save_data["insertion_timestamps"], maxlen=save_data["max_size"])
            self.feature_stats = save_data["feature_stats"]
            self.alpha = save_data["alpha"]
            self.beta = save_data["beta"]
            self.max_size = save_data["max_size"]
            
            logger.info(f"Mémoire de rejeu chargée: {filepath} ({len(self.memory)} exemples)")
            return True
        except Exception as e:
            logger.error(f"Erreur lors du chargement de la mémoire de rejeu: {str(e)}")
            return False

class ConceptDriftDetector:
    """
    Détecteur avancé de concept drift pour identifier les changements dans les données
    et signaler quand le modèle doit être adapté
    """
    def __init__(self, window_size: int = 100, 
                reference_size: int = 500,
                threshold: float = 0.05,
                min_samples: int = 30,
                concept_history_size: int = 10):
        """
        Initialise le détecteur de concept drift
        
        Args:
            window_size: Taille de la fenêtre d'observation
            reference_size: Taille de la fenêtre de référence
            threshold: Seuil de p-valeur pour détecter une dérive
            min_samples: Nombre minimum d'échantillons pour la détection
            concept_history_size: Nombre de concepts précédents à conserver
        """
        self.window_size = window_size
        self.reference_size = reference_size
        self.threshold = threshold
        self.min_samples = min_samples
        
        # Fenêtres de données
        self.reference_window = []
        self.current_window = []
        
        # Historique des drifts détectés
        self.drift_history = []
        
        # Conservation des concepts précédents
        self.concept_history_size = concept_history_size
        self.concept_history = []  # Liste des références précédentes
        
        # Compteur de stabilité pour éviter les faux positifs
        self.stability_counter = 0
        self.required_stability = 3  # Nb de détections positives avant de signaler un drift
        
        # Métadonnées
        self.drift_count = 0
        self.last_drift_time = None
        self.total_observations = 0
    
    def add_observation(self, features: Dict, prediction_error: float, timestamp: str = None) -> Dict:
        """
        Ajoute une observation et vérifie s'il y a une dérive conceptuelle
        
        Args:
            features: Dictionnaire de caractéristiques observées
            prediction_error: Erreur de prédiction associée
            timestamp: Horodatage de l'observation
            
        Returns:
            Résultat de la détection
        """
        # Créer l'observation avec métadonnées
        observation = {
            "features": features,
            "error": prediction_error,
            "timestamp": timestamp or datetime.now().isoformat(),
            "id": self.total_observations
        }
        
        # Incrémenter le compteur total
        self.total_observations += 1
        
        # Ajouter à la fenêtre courante
        self.current_window.append(observation)
        
        # Si la fenêtre courante est trop grande, supprimer les plus anciennes observations
        if len(self.current_window) > self.window_size:
            self.current_window.pop(0)
        
        # Si pas assez d'observations, ou pas de référence, pas de dérive
        if len(self.current_window) < self.min_samples or not self.reference_window:
            return {
                "drift_detected": False,
                "p_value": None,
                "test_statistic": None,
                "message": "Données insuffisantes pour la détection"
            }
        
        # Détecter la dérive
        return self._detect_drift()
    
    def initialize_reference(self, observations: List[Dict] = None) -> None:
        """
        Initialise la fenêtre de référence
        
        Args:
            observations: Liste d'observations pour la référence
        """
        if observations:
            # Utiliser les observations fournies
            self.reference_window = observations[-self.reference_size:] if len(observations) > self.reference_size else observations.copy()
        elif len(self.current_window) >= self.reference_size:
            # Utiliser la fenêtre courante comme référence
            self.reference_window = self.current_window[-self.reference_size:].copy()
        else:
            logger.warning(f"Données insuffisantes pour initialiser la référence ({len(self.current_window)}/{self.reference_size})")
            self.reference_window = self.current_window.copy()
        
        logger.info(f"Fenêtre de référence initialisée avec {len(self.reference_window)} observations")
        
        # Réinitialiser le compteur de stabilité
        self.stability_counter = 0
    
    def update_reference(self) -> None:
        """
        Met à jour la fenêtre de référence avec les données actuelles
        et conserve l'ancienne référence dans l'historique des concepts
        """
        # Sauvegarder la référence actuelle dans l'historique des concepts
        if self.reference_window:
            # Créer un résumé du concept
            concept_summary = self._create_concept_summary(self.reference_window)
            
            # Ajouter à l'historique
            self.concept_history.append({
                "window": self.reference_window.copy(),
                "summary": concept_summary,
                "start_time": self.reference_window[0]["timestamp"] if self.reference_window else None,
                "end_time": datetime.now().isoformat()
            })
            
            # Limiter la taille de l'historique
            if len(self.concept_history) > self.concept_history_size:
                self.concept_history.pop(0)
        
        # Mettre à jour la référence
        self.initialize_reference(self.current_window)
        
        # Enregistrer le drift
        self.drift_count += 1
        self.last_drift_time = datetime.now().isoformat()
        
        # Ajouter aux métadonnées historiques
        self.drift_history.append({
            "timestamp": self.last_drift_time,
            "observation_count": self.total_observations
        })
        
        logger.info(f"Référence mise à jour après détection de drift (#{self.drift_count})")
    
    def _detect_drift(self) -> Dict:
        """
        Détecte si une dérive conceptuelle s'est produite
        
        Returns:
            Résultat de la détection
        """
        # Extraire les erreurs des deux fenêtres
        reference_errors = [obs["error"] for obs in self.reference_window]
        current_errors = [obs["error"] for obs in self.current_window]
        
        # Test statistique pour comparer les distributions (test de Mann-Whitney)
        try:
            stat, p_value = mannwhitneyu(reference_errors, current_errors, alternative='two-sided')
            
            # Test de Kolmogorov-Smirnov en complément
            ks_stat, ks_p_value = ks_2samp(reference_errors, current_errors)
            
            # Considérer qu'il y a dérive si l'une des p-valeurs est inférieure au seuil
            potential_drift = p_value < self.threshold or ks_p_value < self.threshold
            
            if potential_drift:
                self.stability_counter += 1
            else:
                self.stability_counter = max(0, self.stability_counter - 1)  # Réduire le compteur (mais pas en-dessous de 0)
            
            # Dérive confirmée si plusieurs détections consécutives
            drift_detected = self.stability_counter >= self.required_stability
            
            # Caractériser la dérive
            drift_magnitude = None
            drift_direction = None
            
            if drift_detected:
                # Calculer la magnitude et la direction de la dérive
                ref_mean = np.mean(reference_errors)
                cur_mean = np.mean(current_errors)
                
                drift_magnitude = abs(cur_mean - ref_mean) / max(ref_mean, 0.001)
                drift_direction = "worse" if cur_mean > ref_mean else "better"
                
                # Réinitialiser le compteur de stabilité pour les prochaines détections
                self.stability_counter = 0
            
            return {
                "drift_detected": drift_detected,
                "potential_drift": potential_drift,
                "stability_counter": self.stability_counter,
                "p_value": float(p_value),
                "test_statistic": float(stat),
                "ks_p_value": float(ks_p_value),
                "ks_statistic": float(ks_stat),
                "magnitude": float(drift_magnitude) if drift_magnitude is not None else None,
                "direction": drift_direction,
                "reference_size": len(self.reference_window),
                "current_size": len(self.current_window)
            }
        
        except Exception as e:
            logger.error(f"Erreur lors du test de dérive: {str(e)}")
            return {
                "drift_detected": False,
                "error": str(e),
                "message": "Erreur lors du test statistique"
            }
    
    def check_for_concept_return(self) -> Dict:
        """
        Vérifie si les données actuelles correspondent à un concept précédemment observé
        
        Returns:
            Résultat de la vérification
        """
        if not self.concept_history or len(self.current_window) < self.min_samples:
            return {
                "concept_return": False,
                "message": "Historique des concepts vide ou données insuffisantes"
            }
        
        # Extraire les erreurs actuelles
        current_errors = [obs["error"] for obs in self.current_window]
        
        # Tester contre chaque concept historique
        best_match = None
        best_p_value = 0
        
        for i, concept in enumerate(self.concept_history):
            concept_errors = [obs["error"] for obs in concept["window"]]
            
            # Test statistique
            try:
                _, p_value = mannwhitneyu(concept_errors, current_errors, alternative='two-sided')
                
                # Si p-value élevée, les distributions sont similaires
                if p_value > 0.1 and p_value > best_p_value:
                    best_match = i
                    best_p_value = p_value
            except:
                continue
        
        if best_match is not None:
            matched_concept = self.concept_history[best_match]
            
            return {
                "concept_return": True,
                "concept_index": best_match,
                "p_value": float(best_p_value),
                "concept_start_time": matched_concept["start_time"],
                "concept_end_time": matched_concept["end_time"],
                "message": f"Retour au concept #{best_match}"
            }
        
        return {
            "concept_return": False,
            "message": "Aucun concept précédent ne correspond aux données actuelles"
        }
    
    def _create_concept_summary(self, window: List[Dict]) -> Dict:
        """
        Crée un résumé statistique d'un concept
        
        Args:
            window: Fenêtre d'observations
            
        Returns:
            Résumé statistique
        """
        errors = [obs["error"] for obs in window]
        
        # Calculer les statistiques de base
        summary = {
            "error_mean": float(np.mean(errors)),
            "error_std": float(np.std(errors)),
            "error_min": float(np.min(errors)),
            "error_max": float(np.max(errors)),
            "error_median": float(np.median(errors)),
            "sample_count": len(window)
        }
        
        # Extraire des statistiques sur les caractéristiques
        feature_stats = {}
        
        # Obtenir toutes les clés de caractéristiques
        all_keys = set()
        for obs in window:
            all_keys.update(obs["features"].keys())
        
        # Calculer les statistiques pour chaque caractéristique
        for key in all_keys:
            values = [obs["features"].get(key, np.nan) for obs in window]
            values = [v for v in values if not np.isnan(v)]
            
            if values:
                feature_stats[key] = {
                    "mean": float(np.mean(values)),
                    "std": float(np.std(values)),
                    "min": float(np.min(values)),
                    "max": float(np.max(values))
                }
        
        summary["feature_stats"] = feature_stats
        
        return summary
    
    def get_drift_statistics(self) -> Dict:
        """
        Récupère des statistiques sur les drifts détectés
        
        Returns:
            Statistiques de drift
        """
        return {
            "total_drifts": self.drift_count,
            "last_drift_time": self.last_drift_time,
            "drift_history": self.drift_history,
            "total_observations": self.total_observations,
            "concept_history_size": len(self.concept_history)
        }
    
    def save(self, filepath: str) -> None:
        """
        Sauvegarde l'état du détecteur sur disque
        
        Args:
            filepath: Chemin du fichier de sauvegarde
        """
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        save_data = {
            "reference_window": self.reference_window,
            "current_window": self.current_window,
            "drift_history": self.drift_history,
            "concept_history": self.concept_history,
            "drift_count": self.drift_count,
            "last_drift_time": self.last_drift_time,
            "total_observations": self.total_observations,
            "config": {
                "window_size": self.window_size,
                "reference_size": self.reference_size,
                "threshold": self.threshold,
                "min_samples": self.min_samples,
                "concept_history_size": self.concept_history_size
            }
        }
        
        try:
            with open(filepath, 'wb') as f:
                pickle.dump(save_data, f)
            logger.info(f"État du détecteur de concept drift sauvegardé: {filepath}")
        except Exception as e:
            logger.error(f"Erreur lors de la sauvegarde du détecteur: {str(e)}")
    
    def load(self, filepath: str) -> bool:
        """
        Charge l'état du détecteur depuis le disque
        
        Args:
            filepath: Chemin du fichier de sauvegarde
            
        Returns:
            Succès du chargement
        """
        if not os.path.exists(filepath):
            logger.warning(f"Fichier de détecteur non trouvé: {filepath}")
            return False
        
        try:
            with open(filepath, 'rb') as f:
                save_data = pickle.load(f)
            
            self.reference_window = save_data["reference_window"]
            self.current_window = save_data["current_window"]
            self.drift_history = save_data["drift_history"]
            self.concept_history = save_data["concept_history"]
            self.drift_count = save_data["drift_count"]
            self.last_drift_time = save_data["last_drift_time"]
            self.total_observations = save_data["total_observations"]
            
            # Charger la configuration si disponible
            if "config" in save_data:
                config = save_data["config"]
                self.window_size = config.get("window_size", self.window_size)
                self.reference_size = config.get("reference_size", self.reference_size)
                self.threshold = config.get("threshold", self.threshold)
                self.min_samples = config.get("min_samples", self.min_samples)
                self.concept_history_size = config.get("concept_history_size", self.concept_history_size)
            
            logger.info(f"État du détecteur de concept drift chargé: {filepath}")
            return True
        except Exception as e:
            logger.error(f"Erreur lors du chargement du détecteur: {str(e)}")
            return False

class AdvancedContinuousLearning:
    """
    Système d'apprentissage continu avancé qui adapte le modèle aux nouvelles données
    tout en évitant l'oubli catastrophique et en détectant les dérives conceptuelles
    """
    def __init__(self, model,
                 feature_engineering,
                 replay_memory_size: int = 10000,
                 drift_detection_window: int = 100,
                 drift_threshold: float = 0.05,
                 learning_enabled: bool = True,
                 regularization_strength: float = 0.01,
                 elastic_weight_consolidation: bool = True,
                 model_snapshot_interval: int = 5):
        """
        Initialise le système d'apprentissage continu
        
        Args:
            model: Modèle à adapter (LSTM ou autre)
            feature_engineering: Module d'ingénierie des caractéristiques
            replay_memory_size: Taille de la mémoire de rejeu
            drift_detection_window: Taille de la fenêtre pour la détection de drift
            drift_threshold: Seuil de détection de drift
            learning_enabled: Active ou désactive l'apprentissage continu
            regularization_strength: Force de la régularisation pour éviter l'oubli
            elastic_weight_consolidation: Utilise la consolidation élastique des poids
            model_snapshot_interval: Intervalle de prise de snapshots du modèle
        """
        # Composants principaux
        self.model = model
        self.feature_engineering = feature_engineering
        self.learning_enabled = learning_enabled
        self.regularization_strength = regularization_strength
        self.elastic_weight_consolidation = elastic_weight_consolidation
        
        # Mémoire de rejeu avec échantillonnage prioritaire
        self.replay_memory = ReplayMemory(
            max_size=replay_memory_size,
            alpha=0.6,  # Priorité non uniforme (0.6 = modérée)
            beta=0.4    # Correction d'échantillonnage
        )
        
        # Détecteur de concept drift
        self.drift_detector = ConceptDriftDetector(
            window_size=drift_detection_window,
            reference_size=drift_detection_window * 5,
            threshold=drift_threshold,
            min_samples=30,
            concept_history_size=10
        )
        
        # Snapshots du modèle
        self.model_snapshots = []
        self.model_snapshot_interval = model_snapshot_interval
        self.updates_since_snapshot = 0
        
        # Indicateurs d'état
        self.total_updates = 0
        self.last_update_time = None
        self.update_history = []
        
        # Poids importants du modèle (pour EWC)
        self.important_weights = None
        self.fisher_information = None
        
        # Métriques de performance
        self.performance_metrics = {
            "loss_history": [],
            "accuracy_history": []
        }
        
        # Répertoires pour stockage
        self.data_dir = os.path.join(DATA_DIR, "continuous_learning")
        self.replay_memory_path = os.path.join(self.data_dir, "replay_memory.pkl")
        self.drift_detector_path = os.path.join(self.data_dir, "drift_detector.pkl")
        self.snapshots_dir = os.path.join(self.data_dir, "model_snapshots")
        
        # Créer les répertoires
        os.makedirs(self.data_dir, exist_ok=True)
        os.makedirs(self.snapshots_dir, exist_ok=True)
        
        # Charger l'état précédent si disponible
        self._load_state()
    
    def process_new_data(self, data: pd.DataFrame, prediction_errors: List[float] = None,
                        min_samples: int = 30, max_batch_size: int = 64) -> Dict:
        """
        Traite de nouvelles données et met à jour le modèle si nécessaire
        
        Args:
            data: DataFrame avec les nouvelles données OHLCV
            prediction_errors: Erreurs de prédiction associées (optionnel)
            min_samples: Nombre minimum d'échantillons pour la mise à jour
            max_batch_size: Taille maximum des batchs pour la mise à jour
            
        Returns:
            Résultats du traitement
        """
        if not self.learning_enabled:
            return {
                "success": True,
                "updated": False,
                "message": "Apprentissage continu désactivé"
            }
        
        # 1. Prétraiter les données
        try:
            X, y = self._prepare_data(data)
        except Exception as e:
            logger.error(f"Erreur lors de la préparation des données: {str(e)}")
            return {
                "success": False,
                "updated": False,
                "error": str(e)
            }
        
        if len(X) == 0:
            return {
                "success": True,
                "updated": False,
                "message": "Pas de nouvelles données à traiter"
            }
        
        # 2. Ajouter les données à la mémoire de rejeu
        self._add_to_replay_memory(X, y, prediction_errors)
        
        # 3. Vérifier le concept drift
        drift_result = self._check_concept_drift(X, prediction_errors)
        drift_detected = drift_result.get("drift_detected", False)
        
        # 4. Déterminer si une mise à jour est nécessaire
        update_needed = drift_detected or (self.total_updates == 0)
        
        # 5. Mettre à jour le modèle si nécessaire
        if update_needed and len(self.replay_memory) >= min_samples:
            # Avant la mise à jour, si utilisation d'EWC, calculer l'importance des poids actuels
            if self.elastic_weight_consolidation and self.important_weights is None and self.model is not None:
                self._compute_weight_importance()
            
            # Effectuer la mise à jour
            update_result = self._update_model(max_batch_size=max_batch_size)
            
            if update_result["success"]:
                # Après une mise à jour réussie, prendre un snapshot si l'intervalle est atteint
                self.updates_since_snapshot += 1
                
                if self.updates_since_snapshot >= self.model_snapshot_interval:
                    self._create_model_snapshot()
                    self.updates_since_snapshot = 0
                
                # En cas de drift, mettre à jour la référence du détecteur
                if drift_detected:
                    self.drift_detector.update_reference()
                
                # Sauvegarder l'état
                self._save_state()
                
                return {
                    "success": True,
                    "updated": True,
                    "drift_detected": drift_detected,
                    "drift_info": drift_result,
                    "update_result": update_result
                }
            else:
                return {
                    "success": False,
                    "updated": False,
                    "drift_detected": drift_detected,
                    "drift_info": drift_result,
                    "error": update_result.get("error", "Erreur inconnue lors de la mise à jour")
                }
        
        return {
            "success": True,
            "updated": False,
            "drift_detected": drift_detected,
            "drift_info": drift_result,
            "message": "Pas de mise à jour nécessaire ou données insuffisantes"
        }
    
    def _prepare_data(self, data: pd.DataFrame) -> Tuple[np.ndarray, List[np.ndarray]]:
        """
        Prétraite les données pour l'apprentissage continu
        
        Args:
            data: DataFrame avec les données OHLCV
            
        Returns:
            Tuple (X, y) des données prétraitées
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
        
        # Créer les séquences d'entrée et les cibles
        if hasattr(self.model, 'horizon_periods'):
            # Pour le modèle LSTM avancé
            horizons = self.model.horizon_periods
        else:
            # Pour le modèle LSTM standard
            horizons = getattr(self.model, 'prediction_horizons', [12, 24, 96])
        
        # Créer les données avec le bon format
        X, y = self.feature_engineering.create_multi_horizon_data(
            normalized_data,
            sequence_length=getattr(self.model, 'input_length', 60),
            horizons=horizons,
            is_training=True
        )
        
        return X, y
    
    def _add_to_replay_memory(self, X: np.ndarray, y: List[np.ndarray], 
                            prediction_errors: List[float] = None) -> None:
        """
        Ajoute les nouvelles données à la mémoire de rejeu
        
        Args:
            X: Données d'entrée
            y: Données cibles
            prediction_errors: Erreurs de prédiction associées
        """
        # Si pas d'erreurs fournies, utiliser des priorités uniformes
        if prediction_errors is None:
            prediction_errors = [1.0] * len(X)
        
        # S'assurer que nous avons assez d'erreurs
        if len(prediction_errors) < len(X):
            prediction_errors = prediction_errors + [1.0] * (len(X) - len(prediction_errors))
        
        # Ajouter chaque exemple à la mémoire avec sa priorité
        for i in range(len(X)):
            example = (X[i:i+1], [y_arr[i:i+1] for y_arr in y])
            priority = max(0.1, prediction_errors[i])  # Assurer une priorité minimum
            
            self.replay_memory.add(example, priority)
        
        logger.info(f"Ajouté {len(X)} exemples à la mémoire de rejeu (taille: {len(self.replay_memory)})")
    
    def _check_concept_drift(self, X: np.ndarray, prediction_errors: List[float] = None) -> Dict:
        """
        Vérifie s'il y a un concept drift dans les nouvelles données
        
        Args:
            X: Données d'entrée
            prediction_errors: Erreurs de prédiction associées
            
        Returns:
            Résultat de la détection de drift
        """
        # Si pas d'erreurs fournies, utiliser des valeurs par défaut
        if prediction_errors is None or len(prediction_errors) == 0:
            # Faire des prédictions avec le modèle actuel pour obtenir les erreurs
            if self.model is not None:
                try:
                    # Les prédictions dépendent du type de modèle
                    predictions = self.model.model.predict(X)
                    
                    # Calculer l'erreur moyenne pour chaque exemple
                    prediction_errors = []
                    
                    for i in range(len(X)):
                        # Calculer l'erreur moyenne sur tous les horizons et facteurs
                        sample_error = 0.0
                        count = 0
                        
                        for p_idx, pred in enumerate(predictions):
                            if i < len(pred):
                                # MSE pour les sorties numériques, BCE pour les sorties binaires
                                if p_idx % 4 == 0:  # Direction (binaire)
                                    y_true = pred[i][0]
                                    error = -y_true * np.log(max(pred[i][0], 1e-10)) - (1 - y_true) * np.log(max(1 - pred[i][0], 1e-10))
                                else:  # Autres facteurs (régression)
                                    error = (pred[i][0] - y_true) ** 2
                                
                                sample_error += error
                                count += 1
                        
                        if count > 0:
                            prediction_errors.append(sample_error / count)
                        else:
                            prediction_errors.append(1.0)
                
                except Exception as e:
                    logger.error(f"Erreur lors du calcul des erreurs de prédiction: {str(e)}")
                    prediction_errors = [1.0] * len(X)
            else:
                prediction_errors = [1.0] * len(X)
        
        # Extraire des caractéristiques représentatives pour la détection de drift
        drift_features = {}
        
        if len(X) > 0:
            # Moyennes des caractéristiques d'entrée
            feature_means = np.mean(X, axis=(0, 1))
            
            for i, mean in enumerate(feature_means):
                drift_features[f"feature_{i}_mean"] = float(mean)
            
            # Statistiques supplémentaires
            drift_features["input_std"] = float(np.std(X))
            drift_features["input_range"] = float(np.max(X) - np.min(X))
        
        # Détection pour chaque exemple
        drift_detected = False
        drift_details = {
            "observations_checked": len(prediction_errors),
            "drifts_in_window": 0
        }
        
        for i, error in enumerate(prediction_errors):
            # Ajouter l'observation au détecteur
            result = self.drift_detector.add_observation(
                features=drift_features,
                prediction_error=error,
                timestamp=datetime.now().isoformat()
            )
            
            # Vérifier si un drift a été détecté
            if result.get("drift_detected", False):
                drift_detected = True
                drift_details = result
                
                # Un seul drift suffit pour cette itération
                break
        
        return {
            "drift_detected": drift_detected,
            "details": drift_details
        }
    
    def _update_model(self, max_batch_size: int = 64, epochs: int = 5) -> Dict:
        """
        Met à jour le modèle avec les données de la mémoire de rejeu
        
        Args:
            max_batch_size: Taille maximum des batchs d'entraînement
            epochs: Nombre d'époques d'entraînement
            
        Returns:
            Résultat de la mise à jour
        """
        if self.model is None:
            return {
                "success": False,
                "error": "Modèle non initialisé"
            }
        
        # Sauvegarde des poids actuels
        original_weights = self.model.model.get_weights()
        
        # 1. Préparation des callbacks pour l'entraînement
        callbacks = [
            tf.keras.callbacks.EarlyStopping(
                monitor='loss',
                patience=3,
                restore_best_weights=True
            ),
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor='loss',
                factor=0.5,
                patience=2,
                min_lr=1e-6
            )
        ]
        
        # 2. Obtenir un batch équilibré de la mémoire de rejeu
        batch_size = min(max_batch_size, len(self.replay_memory))
        batch_examples = self.replay_memory.get_balanced_batch(
            batch_size=batch_size,
            recency_weight=0.7  # 70% d'exemples récents, 30% d'exemples divers
        )
        
        if not batch_examples:
            return {
                "success": False,
                "error": "Batch vide"
            }
        
        # 3. Préparation des données d'entraînement
        X_train = np.vstack([example[0] for example in batch_examples])
        y_train = []
        
        # Pour chaque sortie, combiner les exemples
        for output_idx in range(len(batch_examples[0][1])):
            y_output = np.vstack([example[1][output_idx] for example in batch_examples])
            y_train.append(y_output)
        
        # 4. Intégration d'EWC si activé
        if self.elastic_weight_consolidation and self.important_weights is not None and self.fisher_information is not None:
            # Créer une fonction de perte personnalisée avec régularisation EWC
            original_loss = self.model.model.loss
            
            def ewc_loss_wrapper(original_loss_fn, lambda_reg, old_params, fisher):
                """Crée une fonction de perte avec régularisation EWC"""
                
                def ewc_loss(y_true, y_pred):
                    # Perte originale
                    loss = original_loss_fn(y_true, y_pred)
                    
                    # Ajout de la régularisation EWC
                    ewc_reg = 0
                    model_params = self.model.model.trainable_weights
                    
                    for i, (p, old_p) in enumerate(zip(model_params, old_params)):
                        f = fisher[i]
                        ewc_reg += tf.reduce_sum(f * tf.square(p - old_p))
                    
                    # Perte totale = perte originale + terme de régularisation
                    return loss + (lambda_reg * ewc_reg)
                
                return ewc_loss
            
            # Appliquer la régularisation EWC
            custom_losses = []
            
            for loss_fn in original_loss:
                custom_losses.append(
                    ewc_loss_wrapper(loss_fn, self.regularization_strength, self.important_weights, self.fisher_information)
                )
            
            # Recompiler le modèle avec les pertes personnalisées
            self.model.model.compile(
                optimizer=self.model.model.optimizer,
                loss=custom_losses,
                metrics=self.model.model.metrics
            )
            
            logger.info("Modèle recompilé avec régularisation EWC")
        
        # 5. Entraînement du modèle
        try:
            history = self.model.model.fit(
                x=X_train,
                y=y_train,
                epochs=epochs,
                batch_size=min(32, len(X_train)),
                callbacks=callbacks,
                verbose=1
            )
            
            # 6. Évaluation des performances
            eval_result = self.model.model.evaluate(X_train, y_train, verbose=0)
            
            # Calculer les métriques agrégées
            if isinstance(eval_result, list):
                avg_loss = np.mean(eval_result[:len(eval_result)//2])  # Première moitié = pertes
                avg_metric = np.mean(eval_result[len(eval_result)//2:])  # Seconde moitié = métriques
            else:
                avg_loss = eval_result
                avg_metric = 0
            
            # Ajouter aux métriques historiques
            self.performance_metrics["loss_history"].append(float(avg_loss))
            self.performance_metrics["accuracy_history"].append(float(avg_metric))
            
            # 7. Vérifier si la mise à jour a dégradé les performances
            if avg_loss > 1.5 * self.performance_metrics["loss_history"][-2] if len(self.performance_metrics["loss_history"]) > 1 else False:
                # Dégradation significative, revenir aux poids originaux
                self.model.model.set_weights(original_weights)
                
                logger.warning(f"Mise à jour annulée: dégradation des performances (perte: {avg_loss:.4f})")
                
                return {
                    "success": False,
                    "reverted": True,
                    "message": "Mise à jour annulée due à une dégradation des performances",
                    "old_loss": self.performance_metrics["loss_history"][-2],
                    "new_loss": avg_loss
                }
            
            # 8. Mise à jour des compteurs
            self.total_updates += 1
            self.last_update_time = datetime.now().isoformat()
            
            self.update_history.append({
                "timestamp": self.last_update_time,
                "samples": len(X_train),
                "loss": float(avg_loss),
                "accuracy": float(avg_metric),
                "epochs": len(history.history["loss"])
            })
            
            # 9. Mettre à jour l'importance des poids si EWC est activé
            if self.elastic_weight_consolidation:
                self._compute_weight_importance()
            
            return {
                "success": True,
                "loss": float(avg_loss),
                "accuracy": float(avg_metric),
                "epochs": len(history.history["loss"]),
                "samples": len(X_train),
                "timestamp": self.last_update_time
            }
            
        except Exception as e:
            # En cas d'erreur, restaurer les poids originaux
            self.model.model.set_weights(original_weights)
            
            logger.error(f"Erreur lors de la mise à jour du modèle: {str(e)}")
            
            return {
                "success": False,
                "error": str(e)
            }
    
    def _compute_weight_importance(self, samples_for_fisher: int = 64) -> None:
        """
        Calcule l'importance des poids actuels pour la consolidation élastique (EWC)
        
        Args:
            samples_for_fisher: Nombre d'échantillons pour estimer la matrice de Fisher
        """
        if len(self.replay_memory) < samples_for_fisher:
            logger.warning(f"Pas assez d'exemples pour calculer l'importance des poids ({len(self.replay_memory)}/{samples_for_fisher})")
            return
        
        # Stocker les poids actuels comme importants
        self.important_weights = self.model.model.get_weights()
        
        # Échantillonner des exemples pour calculer la matrice de Fisher
        batch = self.replay_memory.get_balanced_batch(batch_size=samples_for_fisher)
        
        if not batch:
            logger.warning("Impossible d'obtenir un batch pour le calcul de la matrice de Fisher")
            return
        
        # Préparer les échantillons
        X_samples = np.vstack([example[0] for example in batch])
        y_samples = []
        
        for output_idx in range(len(batch[0][1])):
            y_output = np.vstack([example[1][output_idx] for example in batch])
            y_samples.append(y_output)
        
        # Calculer la matrice de Fisher
        logger.info("Calcul de la matrice de Fisher pour EWC")
        
        try:
            # Version simplifiée de l'estimation de Fisher
            # (Dans une implémentation complète, on calculerait le gradient au carré)
            self.fisher_information = []
            
            for param in self.important_weights:
                # Initialiser avec une petite valeur pour éviter les divisions par zéro
                fisher_diag = np.ones_like(param) * 1e-5
                self.fisher_information.append(fisher_diag)
            
            # Utilisez la magnitude des gradients comme proxy pour l'information de Fisher
            with tf.GradientTape() as tape:
                # Calculer les sorties du modèle
                predictions = self.model.model(X_samples)
                
                # Calculer une perte combinée
                total_loss = 0
                
                for i, y_pred in enumerate(predictions):
                    # Utiliser le bon type de perte selon le type de sortie
                    if i % 4 == 0:  # Direction (binaire)
                        loss = tf.keras.losses.binary_crossentropy(y_samples[i], y_pred)
                    else:  # Autres facteurs (régression)
                        loss = tf.keras.losses.mean_squared_error(y_samples[i], y_pred)
                    
                    total_loss += tf.reduce_mean(loss)
            
            # Calculer les gradients
            gradients = tape.gradient(total_loss, self.model.model.trainable_weights)
            
            # Utiliser le carré des gradients comme approximation de Fisher
            for i, grad in enumerate(gradients):
                if i < len(self.fisher_information):
                    # Prendre le carré des gradients et moyenner sur les échantillons
                    square_grad = tf.square(grad).numpy()
                    self.fisher_information[i] += square_grad
            
            logger.info("Matrice de Fisher calculée avec succès")
            
        except Exception as e:
            logger.error(f"Erreur lors du calcul de la matrice de Fisher: {str(e)}")
            
            # Fallback: utiliser une matrice identité avec une petite valeur
            self.fisher_information = []
            for param in self.important_weights:
                self.fisher_information.append(np.ones_like(param) * 0.1)
    
    def _create_model_snapshot(self) -> None:
        """Crée un snapshot du modèle actuel"""
        if self.model is None:
            return
        
        # Créer un identifiant unique pour le snapshot
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        snapshot_id = f"snapshot_{self.total_updates}_{timestamp}"
        
        # Chemin du snapshot
        snapshot_path = os.path.join(self.snapshots_dir, f"{snapshot_id}.h5")
        
        try:
            # Sauvegarder le modèle
            self.model.model.save(snapshot_path)
            
            # Stocker les métadonnées du snapshot
            snapshot_info = {
                "id": snapshot_id,
                "path": snapshot_path,
                "timestamp": timestamp,
                "update_count": self.total_updates,
                "metrics": {
                    "loss": self.performance_metrics["loss_history"][-1] if self.performance_metrics["loss_history"] else None,
                    "accuracy": self.performance_metrics["accuracy_history"][-1] if self.performance_metrics["accuracy_history"] else None
                }
            }
            
            self.model_snapshots.append(snapshot_info)
            
            # Limiter le nombre de snapshots (garder les 10 plus récents)
            if len(self.model_snapshots) > 10:
                # Supprimer le snapshot le plus ancien
                old_snapshot = self.model_snapshots.pop(0)
                
                if os.path.exists(old_snapshot["path"]):
                    os.remove(old_snapshot["path"])
            
            logger.info(f"Snapshot du modèle créé: {snapshot_path}")
            
            # Sauvegarder les métadonnées des snapshots
            self._save_snapshot_metadata()
            
        except Exception as e:
            logger.error(f"Erreur lors de la création du snapshot: {str(e)}")
    
    def _save_snapshot_metadata(self) -> None:
        """Sauvegarde les métadonnées des snapshots"""
        metadata_path = os.path.join(self.snapshots_dir, "snapshots_metadata.json")
        
        try:
            with open(metadata_path, 'w') as f:
                json.dump(self.model_snapshots, f, indent=2)
        except Exception as e:
            logger.error(f"Erreur lors de la sauvegarde des métadonnées des snapshots: {str(e)}")
    
    def _load_snapshot_metadata(self) -> None:
        """Charge les métadonnées des snapshots"""
        metadata_path = os.path.join(self.snapshots_dir, "snapshots_metadata.json")
        
        if not os.path.exists(metadata_path):
            return
        
        try:
            with open(metadata_path, 'r') as f:
                self.model_snapshots = json.load(f)
        except Exception as e:
            logger.error(f"Erreur lors du chargement des métadonnées des snapshots: {str(e)}")
    
    def restore_from_snapshot(self, snapshot_id: str = None) -> bool:
        """
        Restaure le modèle à partir d'un snapshot
        
        Args:
            snapshot_id: ID du snapshot à restaurer (le plus récent si None)
            
        Returns:
            Succès de la restauration
        """
        if not self.model_snapshots:
            logger.warning("Aucun snapshot disponible")
            return False
        
        # Déterminer le snapshot à utiliser
        if snapshot_id is None:
            # Utiliser le snapshot le plus récent
            snapshot = self.model_snapshots[-1]
        else:
            # Rechercher le snapshot par ID
            snapshot = next((s for s in self.model_snapshots if s["id"] == snapshot_id), None)
            
            if snapshot is None:
                logger.warning(f"Snapshot non trouvé: {snapshot_id}")
                return False
        
        # Vérifier si le fichier existe
        if not os.path.exists(snapshot["path"]):
            logger.warning(f"Fichier de snapshot non trouvé: {snapshot['path']}")
            return False
        
        try:
            # Charger le modèle depuis le snapshot
            self.model.model = tf.keras.models.load_model(snapshot["path"])
            
            logger.info(f"Modèle restauré depuis le snapshot: {snapshot['id']}")
            
            # Réinitialiser EWC après restauration
            self.important_weights = None
            self.fisher_information = None
            
            return True
        except Exception as e:
            logger.error(f"Erreur lors de la restauration du snapshot: {str(e)}")
            return False
    
    def _save_state(self) -> None:
        """Sauvegarde l'état du système d'apprentissage continu"""
        try:
            # Sauvegarder la mémoire de rejeu
            self.replay_memory.save(self.replay_memory_path)
            
            # Sauvegarder le détecteur de concept drift
            self.drift_detector.save(self.drift_detector_path)
            
            # Sauvegarder les métadonnées d'état
            state_metadata = {
                "total_updates": self.total_updates,
                "last_update_time": self.last_update_time,
                "update_history": self.update_history,
                "performance_metrics": self.performance_metrics,
                "learning_enabled": self.learning_enabled,
                "regularization_strength": self.regularization_strength,
                "elastic_weight_consolidation": self.elastic_weight_consolidation,
                "updates_since_snapshot": self.updates_since_snapshot
            }
            
            metadata_path = os.path.join(self.data_dir, "continuous_learning_state.json")
            
            with open(metadata_path, 'w') as f:
                json.dump(state_metadata, f, indent=2, default=str)
            
            logger.info("État du système d'apprentissage continu sauvegardé")
        except Exception as e:
            logger.error(f"Erreur lors de la sauvegarde de l'état: {str(e)}")
    
    def _load_state(self) -> None:
        """Charge l'état du système d'apprentissage continu"""
        try:
            # Charger la mémoire de rejeu
            if os.path.exists(self.replay_memory_path):
                self.replay_memory.load(self.replay_memory_path)
            
            # Charger le détecteur de concept drift
            if os.path.exists(self.drift_detector_path):
                self.drift_detector.load(self.drift_detector_path)
            
            # Charger les métadonnées des snapshots
            self._load_snapshot_metadata()
            
            # Charger les métadonnées d'état
            metadata_path = os.path.join(self.data_dir, "continuous_learning_state.json")
            
            if os.path.exists(metadata_path):
                with open(metadata_path, 'r') as f:
                    state_metadata = json.load(f)
                
                self.total_updates = state_metadata.get("total_updates", 0)
                self.last_update_time = state_metadata.get("last_update_time")
                self.update_history = state_metadata.get("update_history", [])
                self.performance_metrics = state_metadata.get("performance_metrics", {
                    "loss_history": [],
                    "accuracy_history": []
                })
                self.learning_enabled = state_metadata.get("learning_enabled", self.learning_enabled)
                self.regularization_strength = state_metadata.get("regularization_strength", self.regularization_strength)
                self.elastic_weight_consolidation = state_metadata.get("elastic_weight_consolidation", self.elastic_weight_consolidation)
                self.updates_since_snapshot = state_metadata.get("updates_since_snapshot", 0)
                
                logger.info("État du système d'apprentissage continu chargé")
        except Exception as e:
            logger.error(f"Erreur lors du chargement de l'état: {str(e)}")
    
    def get_status(self) -> Dict:
        """
        Récupère l'état courant du système d'apprentissage continu
        
        Returns:
            Dictionnaire avec l'état courant
        """
        return {
            "enabled": self.learning_enabled,
            "total_updates": self.total_updates,
            "last_update_time": self.last_update_time,
            "replay_memory_size": len(self.replay_memory),
            "drift_statistics": self.drift_detector.get_drift_statistics(),
            "snapshots": len(self.model_snapshots),
            "ewc_active": self.elastic_weight_consolidation and self.important_weights is not None,
            "regularization_strength": self.regularization_strength,
            "performance": {
                "current_loss": self.performance_metrics["loss_history"][-1] if self.performance_metrics["loss_history"] else None,
                "loss_trend": "improving" if (len(self.performance_metrics["loss_history"]) > 1 and 
                                            self.performance_metrics["loss_history"][-1] < self.performance_metrics["loss_history"][-2]) else "steady"
            }
        }

def test_system():
    """Fonction pour tester le système d'apprentissage continu"""
    # Créer un modèle de test
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Dense
    
    model = Sequential([
        Dense(10, activation='relu', input_shape=(5,)),
        Dense(5, activation='relu'),
        Dense(1, activation='sigmoid')
    ])
    
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    
    # Créer une classe fictive pour simuler le feature engineering
    class MockFeatureEngineering:
        def create_features(self, data, **kwargs):
            return data
        
        def scale_features(self, data, **kwargs):
            return data
        
        def create_multi_horizon_data(self, data, **kwargs):
            # Simuler des données de séquence
            X = np.random.randn(10, 60, 5)
            y = [np.random.randint(0, 2, (10, 1)) for _ in range(4)]
            return X, y
    
    # Créer le système
    cl_system = AdvancedContinuousLearning(
        model=model,
        feature_engineering=MockFeatureEngineering(),
        replay_memory_size=1000,
        drift_detection_window=50
    )
    
    # Afficher l'état initial
    print("État initial:")
    print(cl_system.get_status())
    
    return cl_system

if __name__ == "__main__":
    test_system()