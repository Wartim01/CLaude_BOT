# utils/model_monitor.py
"""
Module de surveillance des performances du modèle LSTM en temps réel
Détecte la dégradation des performances et alerte lorsque le modèle nécessite un réentraînement
"""
import os
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Union
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import seaborn as sns
import json
from collections import deque

from config.config import DATA_DIR
from ai.models.lstm_model import LSTMModel
from ai.models.feature_engineering import FeatureEngineering
from ai.models.continuous_learning import ConceptDriftDetector
from utils.logger import setup_logger

logger = setup_logger("model_monitor")

class ModelPerformanceTracker:
    """
    Classe pour suivre les performances du modèle LSTM sur le temps
    et détecter toute dégradation qui nécessiterait un réentraînement
    """
    def __init__(self, model: Optional[LSTMModel] = None, 
               feature_engineering: Optional[FeatureEngineering] = None,
               window_size: int = 100,
               performance_threshold: float = 0.65,
               alert_threshold: float = 0.15):
        """
        Initialise le moniteur de performances du modèle
        
        Args:
            model: Instance du modèle LSTM à surveiller
            feature_engineering: Module d'ingénierie des caractéristiques
            window_size: Taille de la fenêtre glissante pour le suivi des performances
            performance_threshold: Seuil de performance minimale acceptable (précision)
            alert_threshold: Seuil de dégradation pour déclencher une alerte
        """
        self.model = model
        self.feature_engineering = feature_engineering or FeatureEngineering()
        self.window_size = window_size
        self.performance_threshold = performance_threshold
        self.alert_threshold = alert_threshold
        
        # Historique des performances
        self.accuracy_history = deque(maxlen=window_size)
        self.prediction_history = deque(maxlen=window_size)
        self.loss_history = deque(maxlen=window_size)
        
        # Métriques par horizon de prédiction
        self.horizon_metrics = {}
        
        # Détecteurs de concept drift pour chaque horizon
        self.drift_detectors = {}
        
        # État actuel du modèle
        self.current_health = "good"  # "good", "degrading", "critical"
        self.last_evaluation_time = None
        self.requires_retraining = False
        
        # Répertoire pour les rapports et visualisations
        self.output_dir = os.path.join(DATA_DIR, "model_monitoring")
        self.alert_history_file = os.path.join(self.output_dir, "alert_history.json")
        
        # Créer les répertoires si nécessaire
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Charger l'historique des alertes
        self.alert_history = self._load_alert_history()
        
        # Initialiser les détecteurs de concept drift
        self._initialize_drift_detectors()
    
    def _initialize_drift_detectors(self) -> None:
        """
        Initialise les détecteurs de concept drift pour chaque horizon
        """
        if self.model:
            # Obtenir les horizons de prédiction
            prediction_horizons = getattr(self.model, 'prediction_horizons', [12, 24, 96])
            
            # Créer un détecteur pour chaque horizon
            for horizon in prediction_horizons:
                horizon_key = f"horizon_{horizon}"
                self.drift_detectors[horizon_key] = ConceptDriftDetector(
                    window_size=50,
                    reference_size=200,
                    threshold=0.05,
                    min_samples=30
                )
                
                # Initialiser les métriques pour cet horizon
                self.horizon_metrics[horizon_key] = {
                    "accuracy": deque(maxlen=self.window_size),
                    "direction_tp": deque(maxlen=self.window_size),
                    "direction_fp": deque(maxlen=self.window_size),
                    "direction_tn": deque(maxlen=self.window_size),
                    "direction_fn": deque(maxlen=self.window_size),
                    "volatility_mae": deque(maxlen=self.window_size),
                }
    
    def track_prediction(self, prediction: Dict, actual_values: Dict, 
                       prediction_time: datetime = None) -> Dict:
        """
        Enregistre une prédiction et sa performance réelle
        
        Args:
            prediction: Prédiction faite par le modèle
            actual_values: Valeurs réelles observées après l'horizon de prédiction
            prediction_time: Horodatage de la prédiction
            
        Returns:
            Métriques de performance pour cette prédiction
        """
        if prediction_time is None:
            prediction_time = datetime.now()
        
        # Calculer les métriques de performance
        performance_metrics = self._calculate_performance_metrics(prediction, actual_values)
        
        # Ajouter à l'historique
        self.accuracy_history.append(performance_metrics["overall_accuracy"])
        self.loss_history.append(performance_metrics["overall_loss"])
        
        # Stocker la prédiction avec ses métriques
        prediction_record = {
            "timestamp": prediction_time.isoformat(),
            "prediction": prediction,
            "actual_values": actual_values,
            "metrics": performance_metrics
        }
        self.prediction_history.append(prediction_record)
        
        # Mettre à jour les métriques par horizon
        for horizon_key, horizon_metrics in performance_metrics["horizons"].items():
            if horizon_key in self.horizon_metrics:
                self.horizon_metrics[horizon_key]["accuracy"].append(horizon_metrics["direction_accuracy"])
                self.horizon_metrics[horizon_key]["volatility_mae"].append(horizon_metrics.get("volatility_mae", 0))
                
                # Mettre à jour les statistiques pour la matrice de confusion
                self.horizon_metrics[horizon_key]["direction_tp"].append(horizon_metrics.get("tp", 0))
                self.horizon_metrics[horizon_key]["direction_fp"].append(horizon_metrics.get("fp", 0))
                self.horizon_metrics[horizon_key]["direction_tn"].append(horizon_metrics.get("tn", 0))
                self.horizon_metrics[horizon_key]["direction_fn"].append(horizon_metrics.get("fn", 0))
        
        # Vérifier le concept drift pour chaque horizon
        drift_detected = False
        drifting_horizons = []
        
        for horizon_key, detector in self.drift_detectors.items():
            if horizon_key in performance_metrics["horizons"]:
                error = 1.0 - performance_metrics["horizons"][horizon_key]["direction_accuracy"]
                
                drift_result = detector.add_observation(
                    features={"error": error},
                    prediction_error=error,
                    timestamp=prediction_time.isoformat()
                )
                
                if drift_result.get("drift_detected", False):
                    drift_detected = True
                    drifting_horizons.append(horizon_key)
                    logger.warning(f"Concept drift détecté pour {horizon_key}")
        
        # Mettre à jour l'état de santé du modèle
        self._update_model_health()
        
        # Enregistrer l'évaluation
        self.last_evaluation_time = prediction_time
        
        # Créer périodiquement des visualisations et rapports
        if len(self.accuracy_history) % 20 == 0:  # Tous les 20 échantillons
            self.generate_performance_report()
        
        # Générer une alerte si nécessaire
        alert = None
        if drift_detected or self.current_health == "critical":
            alert = self._generate_alert(drifting_horizons)
        
        return {
            "metrics": performance_metrics,
            "current_health": self.current_health,
            "drift_detected": drift_detected,
            "drifting_horizons": drifting_horizons,
            "requires_retraining": self.requires_retraining,
            "alert": alert
        }
    
    def _calculate_performance_metrics(self, prediction: Dict, actual_values: Dict) -> Dict:
        """
        Calcule les métriques de performance pour une prédiction
        
        Args:
            prediction: Prédiction faite par le modèle
            actual_values: Valeurs réelles observées
            
        Returns:
            Métriques de performance
        """
        performance_metrics = {
            "horizons": {},
            "overall_accuracy": 0.0,
            "overall_loss": 0.0,
            "samples_count": 1
        }
        
        # Pour chaque horizon, calculer les métriques
        correct_count = 0
        total_count = 0
        total_loss = 0.0
        
        for horizon_key, horizon_pred in prediction.items():
            if horizon_key == "reversal_alert":
                continue  # Ignorer l'alerte de retournement
                
            if horizon_key not in actual_values:
                continue  # Ignorer les horizons sans valeurs réelles
            
            actual = actual_values[horizon_key]
            
            # Calculer les métriques pour la direction
            predicted_direction = horizon_pred.get("direction_probability", 50) > 50
            actual_direction = actual.get("actual_direction", 0) > 0
            
            direction_correct = predicted_direction == actual_direction
            
            # Mettre à jour les compteurs
            correct_count += 1 if direction_correct else 0
            total_count += 1
            
            # Calculer les métriques pour la direction avec matrice de confusion
            tp = 1 if predicted_direction and actual_direction else 0
            fp = 1 if predicted_direction and not actual_direction else 0
            tn = 1 if not predicted_direction and not actual_direction else 0
            fn = 1 if not predicted_direction and actual_direction else 0
            
            direction_accuracy = 1.0 if direction_correct else 0.0
            
            # Calculer la précision, le rappel et le F1-score
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1_score = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
            
            # Calculer l'erreur logarithmique pour la direction
            pred_prob = horizon_pred.get("direction_probability", 50) / 100.0
            actual_val = 1.0 if actual_direction else 0.0
            
            # Log loss (pour éviter log(0), utiliser un epsilon)
            epsilon = 1e-15
            pred_prob = max(epsilon, min(1 - epsilon, pred_prob))
            log_loss = -actual_val * np.log(pred_prob) - (1 - actual_val) * np.log(1 - pred_prob)
            
            # Calculer l'erreur pour la volatilité
            volatility_error = 0.0
            if "predicted_volatility" in horizon_pred and "actual_volatility" in actual:
                pred_volatility = horizon_pred["predicted_volatility"]
                actual_volatility = actual["actual_volatility"]
                volatility_error = abs(pred_volatility - actual_volatility)
            
            # Enregistrer les métriques pour cet horizon
            performance_metrics["horizons"][horizon_key] = {
                "direction_accuracy": direction_accuracy,
                "direction_precision": precision,
                "direction_recall": recall,
                "direction_f1": f1_score,
                "log_loss": log_loss,
                "volatility_mae": volatility_error,
                "tp": tp,
                "fp": fp,
                "tn": tn,
                "fn": fn
            }
            
            # Mettre à jour la perte totale
            total_loss += log_loss
        
        # Calculer les métriques globales
        if total_count > 0:
            performance_metrics["overall_accuracy"] = correct_count / total_count
            performance_metrics["overall_loss"] = total_loss / total_count
        
        return performance_metrics
    
    def _update_model_health(self) -> None:
        """
        Met à jour l'état de santé du modèle en fonction des performances récentes
        """
        if len(self.accuracy_history) < 10:
            # Pas assez de données pour évaluer
            self.current_health = "monitoring"
            self.requires_retraining = False
            return
        
        # Calculer la précision moyenne sur les dernières prédictions
        recent_accuracy = np.mean(list(self.accuracy_history)[-10:])
        
        # Calculer la tendance de précision (pente)
        if len(self.accuracy_history) >= 20:
            accuracy_array = np.array(list(self.accuracy_history)[-20:])
            x = np.arange(len(accuracy_array))
            slope = np.polyfit(x, accuracy_array, 1)[0]
        else:
            slope = 0
        
        # Évaluer l'état de santé
        if recent_accuracy < self.performance_threshold:
            if recent_accuracy < self.performance_threshold - self.alert_threshold:
                self.current_health = "critical"
                self.requires_retraining = True
            else:
                self.current_health = "degrading"
                self.requires_retraining = slope < -0.01  # Tendance négative significative
        else:
            if slope < -0.02:  # Forte tendance négative même au-dessus du seuil
                self.current_health = "degrading"
                self.requires_retraining = False
            else:
                self.current_health = "good"
                self.requires_retraining = False
    
    def _generate_alert(self, drifting_horizons: List[str]) -> Dict:
        """
        Génère une alerte en cas de problème de performance
        
        Args:
            drifting_horizons: Liste des horizons présentant un concept drift
            
        Returns:
            Informations sur l'alerte
        """
        alert = {
            "timestamp": datetime.now().isoformat(),
            "model_health": self.current_health,
            "requires_retraining": self.requires_retraining,
            "drift_detected": len(drifting_horizons) > 0,
            "drifting_horizons": drifting_horizons,
            "recent_accuracy": np.mean(list(self.accuracy_history)[-10:]) if len(self.accuracy_history) >= 10 else None,
            "details": self._get_alert_details()
        }
        
        # Ajouter l'alerte à l'historique
        self.alert_history.append(alert)
        
        # Sauvegarder l'historique des alertes
        self._save_alert_history()
        
        # Générer un rapport pour cette alerte
        self._generate_alert_report(alert)
        
        return alert
    
    def _get_alert_details(self) -> Dict:
        """
        Récupère les détails supplémentaires pour une alerte
        
        Returns:
            Détails de l'alerte
        """
        details = {
            "horizon_performance": {}
        }
        
        # Récupérer les performances récentes par horizon
        for horizon_key, metrics in self.horizon_metrics.items():
            if len(metrics["accuracy"]) > 0:
                recent_accuracy = np.mean(list(metrics["accuracy"])[-10:])
                
                # Calculer la tendance
                if len(metrics["accuracy"]) >= 20:
                    accuracy_array = np.array(list(metrics["accuracy"])[-20:])
                    x = np.arange(len(accuracy_array))
                    slope = np.polyfit(x, accuracy_array, 1)[0]
                else:
                    slope = 0
                
                details["horizon_performance"][horizon_key] = {
                    "recent_accuracy": recent_accuracy,
                    "trend_slope": slope,
                    "status": "degrading" if slope < -0.01 or recent_accuracy < self.performance_threshold else "stable"
                }
        
        return details
    
    def reset_drift_detection(self) -> None:
        """
        Réinitialise les détecteurs de concept drift après réentraînement
        """
        # Réinitialiser les détecteurs
        for horizon_key, detector in self.drift_detectors.items():
            detector.initialize_reference()
            logger.info(f"Détecteur de drift réinitialisé pour {horizon_key}")
        
        # Marquer le modèle comme en bonne santé
        self.current_health = "good"
        self.requires_retraining = False
        
        logger.info("Surveillance du modèle réinitialisée après réentraînement")
    
    def generate_performance_report(self, detailed: bool = True) -> str:
        """
        Génère un rapport de performance avec des visualisations
        
        Args:
            detailed: Générer un rapport détaillé
            
        Returns:
            Chemin vers le rapport généré
        """
        if len(self.accuracy_history) < 10:
            logger.warning("Pas assez de données pour générer un rapport")
            return None
        
        # Créer le répertoire pour les rapports
        reports_dir = os.path.join(self.output_dir, "reports")
        os.makedirs(reports_dir, exist_ok=True)
        
        # 1. Créer un rapport au format JSON
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_file = os.path.join(reports_dir, f"performance_report_{timestamp}.json")
        
        # Préparer les données du rapport
        report_data = {
            "timestamp": datetime.now().isoformat(),
            "model_health": self.current_health,
            "requires_retraining": self.requires_retraining,
            "overall_metrics": {
                "recent_accuracy": np.mean(list(self.accuracy_history)[-10:]),
                "recent_loss": np.mean(list(self.loss_history)[-10:]),
                "accuracy_trend": self._calculate_trend(self.accuracy_history),
                "total_samples": len(self.accuracy_history)
            },
            "horizon_metrics": {}
        }
        
        # Ajouter les métriques par horizon
        for horizon_key, metrics in self.horizon_metrics.items():
            if len(metrics["accuracy"]) > 0:
                # Calculer les métriques sur les 10 derniers échantillons
                recent_metrics = {
                    "accuracy": np.mean(list(metrics["accuracy"])[-10:]),
                    "accuracy_trend": self._calculate_trend(metrics["accuracy"]),
                    "volatility_mae": np.mean(list(metrics["volatility_mae"])[-10:]),
                    
                    # Métriques de confusion
                    "tp": np.sum(list(metrics["direction_tp"])[-10:]),
                    "fp": np.sum(list(metrics["direction_fp"])[-10:]),
                    "tn": np.sum(list(metrics["direction_tn"])[-10:]),
                    "fn": np.sum(list(metrics["direction_fn"])[-10:])
                }
                
                # Calculer la précision, le rappel et le F1-score
                tp = recent_metrics["tp"]
                fp = recent_metrics["fp"]
                tn = recent_metrics["tn"]
                fn = recent_metrics["fn"]
                
                precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
                recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
                f1_score = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
                
                recent_metrics["precision"] = precision
                recent_metrics["recall"] = recall
                recent_metrics["f1_score"] = f1_score
                
                report_data["horizon_metrics"][horizon_key] = recent_metrics
        
        # Sauvegarder le rapport
        with open(report_file, 'w') as f:
            json.dump(report_data, f, indent=2)
        
        # 2. Générer des visualisations si demandé
        if detailed:
            # Générer les visualisations
            viz_file = self._generate_performance_visualizations(timestamp)
            report_data["visualizations"] = viz_file
        
        logger.info(f"Rapport de performance généré: {report_file}")
        return report_file
    
    def _generate_performance_visualizations(self, timestamp: str) -> str:
        """
        Génère des visualisations de performance
        
        Args:
            timestamp: Horodatage pour les noms de fichiers
            
        Returns:
            Chemin vers le fichier de visualisations
        """
        # Créer le répertoire pour les visualisations
        viz_dir = os.path.join(self.output_dir, "visualizations")
        os.makedirs(viz_dir, exist_ok=True)
        
        # Nom du fichier de visualisation
        viz_file = os.path.join(viz_dir, f"performance_viz_{timestamp}.png")
        
        # 1. Créer un graphique avec plusieurs sous-graphiques
        fig, axs = plt.subplots(3, 1, figsize=(14, 18), gridspec_kw={'height_ratios': [2, 1, 2]})
        
        # Configurer le style
        plt.style.use('ggplot')
        
        # 2. Premier sous-graphique: Précision globale au fil du temps
        ax1 = axs[0]
        
        # Convertir la queue en liste pour le tracé
        accuracy_values = list(self.accuracy_history)
        x = range(len(accuracy_values))
        
        # Tracer la précision
        ax1.plot(x, accuracy_values, '-', color='blue', label='Précision globale')
        
        # Ajouter une ligne de tendance
        if len(accuracy_values) > 5:
            z = np.polyfit(x, accuracy_values, 1)
            p = np.poly1d(z)
            ax1.plot(x, p(x), "r--", alpha=0.7, label='Tendance')
        
        # Ajouter des lignes de seuil
        ax1.axhline(y=self.performance_threshold, color='orange', linestyle='--', 
                   alpha=0.7, label=f'Seuil minimal ({self.performance_threshold})')
        
        ax1.axhline(y=self.performance_threshold - self.alert_threshold, color='red', 
                   linestyle='--', alpha=0.7, label=f'Seuil critique ({self.performance_threshold - self.alert_threshold})')
        
        # Configurer le graphique
        ax1.set_title('Évolution de la précision globale du modèle', fontsize=14)
        ax1.set_xlabel('Échantillons')
        ax1.set_ylabel('Précision')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Ajouter une annotation pour l'état de santé
        health_color = {
            "good": "green",
            "degrading": "orange",
            "critical": "red",
            "monitoring": "blue"
        }.get(self.current_health, "gray")
        
        retraining_msg = "Réentraînement recommandé" if self.requires_retraining else ""
        
        ax1.text(0.02, 0.05, f"État: {self.current_health.upper()}\n{retraining_msg}", 
                transform=ax1.transAxes, fontsize=12, fontweight='bold',
                bbox=dict(facecolor='white', alpha=0.7, edgecolor=health_color, boxstyle='round,pad=0.5'),
                color=health_color)
        
        # 3. Deuxième sous-graphique: Perte globale au fil du temps
        ax2 = axs[1]
        
        # Convertir la queue en liste pour le tracé
        loss_values = list(self.loss_history)
        x = range(len(loss_values))
        
        # Tracer la perte
        ax2.plot(x, loss_values, '-', color='purple', label='Perte globale')
        
        # Ajouter une ligne de tendance
        if len(loss_values) > 5:
            z = np.polyfit(x, loss_values, 1)
            p = np.poly1d(z)
            ax2.plot(x, p(x), "r--", alpha=0.7, label='Tendance')
        
        # Configurer le graphique
        ax2.set_title('Évolution de la perte globale du modèle', fontsize=14)
        ax2.set_xlabel('Échantillons')
        ax2.set_ylabel('Perte')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # 4. Troisième sous-graphique: Précision par horizon
        ax3 = axs[2]
        
        # Pour chaque horizon, tracer la précision
        for horizon_key, metrics in self.horizon_metrics.items():
            if len(metrics["accuracy"]) > 0:
                accuracy_values = list(metrics["accuracy"])
                x = range(len(accuracy_values))
                
                # Raccourcir le nom de l'horizon pour la lisibilité
                label = horizon_key.replace("horizon_", "h")
                
                ax3.plot(x, accuracy_values, '-', label=label)
        
        # Configurer le graphique
        ax3.set_title('Précision par horizon de prédiction', fontsize=14)
        ax3.set_xlabel('Échantillons')
        ax3.set_ylabel('Précision')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # 5. Configurer le graphique global
        plt.tight_layout()
        
        # Sauvegarder le graphique
        plt.savefig(viz_file)
        plt.close(fig)
        
        return viz_file
    
    def _generate_alert_report(self, alert: Dict) -> str:
        """
        Génère un rapport spécifique pour une alerte
        
        Args:
            alert: Données de l'alerte
            
        Returns:
            Chemin vers le rapport d'alerte
        """
        # Créer le répertoire pour les alertes
        alerts_dir = os.path.join(self.output_dir, "alerts")
        os.makedirs(alerts_dir, exist_ok=True)
        
        # Nom du fichier d'alerte
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        alert_file = os.path.join(alerts_dir, f"alert_{timestamp}.json")
        
        # Sauvegarder l'alerte
        with open(alert_file, 'w') as f:
            json.dump(alert, f, indent=2)
        
        # Générer une visualisation pour l'alerte
        viz_file = os.path.join(alerts_dir, f"alert_viz_{timestamp}.png")
        
        # Créer un graphique pour l'alerte
        plt.figure(figsize=(12, 8))
        
        # Convertir la queue en liste pour le tracé
        accuracy_values = list(self.accuracy_history)
        x = range(len(accuracy_values))
        
        # Tracer la précision
        plt.plot(x, accuracy_values, '-', color='blue', label='Précision globale')
        
        # Marquer le point d'alerte
        plt.plot([len(accuracy_values) - 1], [accuracy_values[-1]], 'ro', markersize=10, 
                label='Point d\'alerte')
        
        # Ajouter des lignes de seuil
        plt.axhline(y=self.performance_threshold, color='orange', linestyle='--', 
                   alpha=0.7, label=f'Seuil minimal ({self.performance_threshold})')
        
        plt.axhline(y=self.performance_threshold - self.alert_threshold, color='red', 
                   linestyle='--', alpha=0.7, label=f'Seuil critique ({self.performance_threshold - self.alert_threshold})')
        
        # Configurer le graphique
        title = "ALERTE: " + (
            "Dérive conceptuelle détectée" if alert["drift_detected"] else 
            "Performances critiques" if alert["model_health"] == "critical" else
            "Dégradation des performances" if alert["model_health"] == "degrading" else
            "Problème de performance"
        )
        plt.title(title, fontsize=16, color='red', fontweight='bold')
        plt.xlabel('Échantillons')
        plt.ylabel('Précision')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Ajouter un texte d'alerte
        alert_text = (
            f"État: {alert['model_health'].upper()}\n"
            f"Précision récente: {alert['recent_accuracy']:.3f}\n"
            f"Réentraînement nécessaire: {'OUI' if alert['requires_retraining'] else 'Non'}\n"
            f"Horizons en dérive: {', '.join(alert['drifting_horizons']) if alert['drifting_horizons'] else 'Aucun'}"
        )
        plt.text(0.02, 0.05, alert_text, transform=plt.gca().transAxes, fontsize=12,
                bbox=dict(facecolor='white', alpha=0.7, edgecolor='red', boxstyle='round,pad=0.5'))
        
        # Sauvegarder le graphique
        plt.tight_layout()
        plt.savefig(viz_file)
        plt.close()
        
        return alert_file
    
    def _calculate_trend(self, data_queue: deque) -> float:
        """
        Calcule la tendance (pente) d'une série de données
        
        Args:
            data_queue: File de données
            
        Returns:
            Pente de la tendance
        """
        if len(data_queue) < 5:
            return 0.0
        
        data_list = list(data_queue)
        x = np.arange(len(data_list))
        
        # Calculer la régression linéaire
        try:
            slope, _ = np.polyfit(x, data_list, 1)
            return float(slope)
        except:
            return 0.0
    
    def _load_alert_history(self) -> List[Dict]:
        """
        Charge l'historique des alertes depuis le fichier
        
        Returns:
            Liste des alertes historiques
        """
        if os.path.exists(self.alert_history_file):
            try:
                with open(self.alert_history_file, 'r') as f:
                    return json.load(f)
            except:
                logger.warning("Erreur lors du chargement de l'historique des alertes")
        
        return []
    
    def _save_alert_history(self) -> None:
        """
        Sauvegarde l'historique des alertes dans un fichier
        """
        # Limiter la taille de l'historique des alertes
        if len(self.alert_history) > 100:
            self.alert_history = self.alert_history[-100:]
            
        try:
            with open(self.alert_history_file, 'w') as f:
                json.dump(self.alert_history, f, indent=2)
        except:
            logger.warning("Erreur lors de la sauvegarde de l'historique des alertes")
    
    def get_model_status(self) -> Dict:
        """
        Récupère l'état actuel du modèle et ses métriques de performance
        
        Returns:
            État du modèle et métriques de performance
        """
        # Calculer les métriques récentes
        recent_accuracy = np.mean(list(self.accuracy_history)[-10:]) if len(self.accuracy_history) >= 10 else None
        recent_loss = np.mean(list(self.loss_history)[-10:]) if len(self.loss_history) >= 10 else None
        
        # Calculer les tendances
        accuracy_trend = self._calculate_trend(self.accuracy_history)
        loss_trend = self._calculate_trend(self.loss_history)
        
        # Récupérer les métriques par horizon
        horizon_status = {}
        for horizon_key, metrics in self.horizon_metrics.items():
            if len(metrics["accuracy"]) >= 10:
                recent_horizon_accuracy = np.mean(list(metrics["accuracy"])[-10:])
                horizon_trend = self._calculate_trend(metrics["accuracy"])
                
                horizon_status[horizon_key] = {
                    "accuracy": recent_horizon_accuracy,
                    "trend": horizon_trend,
                    "health": "good" if recent_horizon_accuracy >= self.performance_threshold else
                             "critical" if recent_horizon_accuracy < self.performance_threshold - self.alert_threshold else
                             "degrading"
                }
        
        # Compiler le statut global
        status = {
            "model_health": self.current_health,
            "requires_retraining": self.requires_retraining,
            "overall_metrics": {
                "accuracy": recent_accuracy,
                "loss": recent_loss,
                "accuracy_trend": accuracy_trend,
                "loss_trend": loss_trend
            },
            "horizon_metrics": horizon_status,
            "samples_count": len(self.accuracy_history),
            "last_evaluation": self.last_evaluation_time.isoformat() if self.last_evaluation_time else None,
            "alert_count": len(self.alert_history)
        }
        
        return status


class ModelMonitor:
    """
    Classe principale pour le monitoring des modèles en production
    Intègre le suivi des performances et la détection des anomalies
    """
    def __init__(self, model_path: Optional[str] = None,
               feature_engineering: Optional[FeatureEngineering] = None,
               enable_alerts: bool = True,
               alert_threshold: float = 0.15,
               check_interval: int = 24):  # Heures
        """
        Initialise le moniteur de modèle
        
        Args:
            model_path: Chemin vers le modèle à surveiller
            feature_engineering: Module d'ingénierie des caractéristiques
            enable_alerts: Activer les alertes automatiques
            alert_threshold: Seuil de dégradation pour les alertes
            check_interval: Intervalle entre les vérifications complètes (heures)
        """
        # Charger le modèle si un chemin est fourni
        self.model = None
        if model_path:
            try:
                self.model = LSTMModel()
                self.model.load(model_path)
                logger.info(f"Modèle chargé pour monitoring: {model_path}")
            except Exception as e:
                logger.error(f"Erreur lors du chargement du modèle: {str(e)}")
        
        # Initialiser le feature engineering
        self.feature_engineering = feature_engineering or FeatureEngineering()
        
        # Paramètres de surveillance
        self.enable_alerts = enable_alerts
        self.alert_threshold = alert_threshold
        self.check_interval = check_interval
        
        # Tracker de performances
        self.performance_tracker = ModelPerformanceTracker(
            model=self.model,
            feature_engineering=self.feature_engineering,
            alert_threshold=alert_threshold
        )
        
        # Historique des prédictions récentes pour calcul des métriques
        self.prediction_cache = {}
        
        # Horodatage de la dernière vérification complète
        self.last_full_check = None
        
        # Répertoire pour les rapports
        self.output_dir = os.path.join(DATA_DIR, "model_monitoring")
        os.makedirs(self.output_dir, exist_ok=True)
    
    def record_prediction(self, symbol: str, prediction: Dict, 
                         actual_data: Optional[pd.DataFrame] = None) -> None:
        """
        Enregistre une prédiction pour le suivi des performances
        
        Args:
            symbol: Symbole de trading
            prediction: Prédiction du modèle
            actual_data: Données réelles pour valider la prédiction (si disponibles)
        """
        # Enregistrer la prédiction avec horodatage
        timestamp = datetime.now()
        
        prediction_record = {
            "timestamp": timestamp,
            "prediction": prediction,
            "symbol": symbol,
            "validated": False
        }
        
        # Stocker dans le cache avec une clé unique
        cache_key = f"{symbol}_{timestamp.strftime('%Y%m%d_%H%M%S')}"
        self.prediction_cache[cache_key] = prediction_record
        
        # Si des données réelles sont fournies, valider immédiatement
        if actual_data is not None:
            self.validate_prediction(cache_key, actual_data)
        
        # Limiter la taille du cache
        if len(self.prediction_cache) > 1000:
            # Supprimer les entrées les plus anciennes
            cache_keys = sorted(self.prediction_cache.keys(), 
                               key=lambda k: self.prediction_cache[k]["timestamp"])
            
            for key in cache_keys[:500]:  # Garder seulement les 500 plus récentes
                if self.prediction_cache[key]["validated"]:  # Ne supprimer que les validées
                    del self.prediction_cache[key]
        
        # Vérifier s'il faut faire une vérification complète
        self._check_full_verification_needed()
    
    def validate_prediction(self, prediction_key: str, actual_data: pd.DataFrame) -> Dict:
        """
        Valide une prédiction précédente avec les données réelles
        
        Args:
            prediction_key: Clé de la prédiction à valider
            actual_data: Données réelles observées
            
        Returns:
            Résultats de validation
        """
        if prediction_key not in self.prediction_cache:
            logger.warning(f"Prédiction non trouvée: {prediction_key}")
            return {"success": False, "error": "Prédiction non trouvée"}
        
        # Récupérer la prédiction
        prediction_record = self.prediction_cache[prediction_key]
        prediction = prediction_record["prediction"]
        
        # Si déjà validée, retourner les résultats existants
        if prediction_record.get("validated", False):
            return prediction_record.get("validation_results", {"success": False, "error": "Données de validation manquantes"})
        
        # Extraire les valeurs réelles pour chaque horizon
        actual_values = self._extract_actual_values(prediction, actual_data)
        
        # Passer au tracker de performances
        track_results = self.performance_tracker.track_prediction(
            prediction=prediction,
            actual_values=actual_values,
            prediction_time=prediction_record["timestamp"]
        )
        
        # Mettre à jour le cache avec les résultats de validation
        prediction_record["validated"] = True
        prediction_record["validation_results"] = track_results
        prediction_record["actual_values"] = actual_values
        
        # Gérer les alertes si nécessaire
        if self.enable_alerts and track_results.get("alert"):
            self._handle_alert(track_results["alert"])
        
        return track_results
    
    def _extract_actual_values(self, prediction: Dict, actual_data: pd.DataFrame) -> Dict:
        """
        Extrait les valeurs réelles correspondant aux prédictions
        
        Args:
            prediction: Prédiction originale
            actual_data: Données réelles
            
        Returns:
            Valeurs réelles par horizon
        """
        actual_values = {}
        
        # Pour chaque horizon de prédiction
        for horizon_key, horizon_pred in prediction.items():
            if horizon_key == "reversal_alert":
                continue  # Ignorer l'alerte de retournement
            
            # Extraire l'horizon numérique
            horizon = int(horizon_key.replace("horizon_", "")) if "horizon_" in horizon_key else 12
            
            # Les données historiques doivent contenir au moins cet horizon
            if len(actual_data) < horizon:
                continue
            
            # Prix actuel et futur
            current_price = actual_data['close'].iloc[0]
            future_price = actual_data['close'].iloc[horizon - 1]
            
            # Direction réelle
            actual_direction = future_price > current_price
            
            # Volatilité réelle (écart-type des rendements)
            returns = actual_data['close'].pct_change().dropna().iloc[:horizon]
            actual_volatility = returns.std() * np.sqrt(horizon)
            
            # Momentum réel
            price_change_pct = (future_price - current_price) / current_price
            actual_momentum = np.tanh(price_change_pct * 5)  # Normaliser entre -1 et 1
            
            # Volume relatif
            if 'volume' in actual_data.columns:
                current_volume = actual_data['volume'].iloc[0]
                future_volume = actual_data['volume'].iloc[:horizon].mean()
                actual_volume_ratio = future_volume / current_volume if current_volume > 0 else 1.0
            else:
                actual_volume_ratio = 1.0
            
            # Stocker les valeurs réelles
            actual_values[horizon_key] = {
                "actual_direction": 1 if actual_direction else 0,
                "actual_volatility": actual_volatility,
                "actual_momentum": actual_momentum,
                "actual_volume_ratio": actual_volume_ratio,
                "price_change_pct": price_change_pct * 100  # En pourcentage
            }
        
        return actual_values
    
    def _handle_alert(self, alert: Dict) -> None:
        """
        Gère une alerte de performance
        
        Args:
            alert: Informations sur l'alerte
        """
        logger.warning(f"ALERTE DE PERFORMANCE: {alert.get('model_health', 'unknown')}")
        
        # Ici, vous pouvez implémenter différentes actions en fonction de l'alerte:
        # - Envoyer un email
        # - Déclencher un réentraînement automatique
        # - Écrire dans un journal d'alertes
        # - etc.
        
        # Par exemple, générer un rapport de performance
        self.performance_tracker.generate_performance_report(detailed=True)
        
        # Notifier dans les logs
        if alert.get("requires_retraining", False):
            logger.critical("RÉENTRAINEMENT DU MODÈLE NÉCESSAIRE")
        elif alert.get("drift_detected", False):
            logger.warning(f"Concept drift détecté: {', '.join(alert.get('drifting_horizons', []))}")
    
    def _check_full_verification_needed(self) -> None:
        """
        Vérifie s'il faut effectuer une vérification complète du modèle
        """
        current_time = datetime.now()
        
        # Si pas de vérification précédente ou intervalle dépassé
        if (self.last_full_check is None or 
            (current_time - self.last_full_check).total_seconds() / 3600 >= self.check_interval):
            
            # Générer un rapport complet
            self.performance_tracker.generate_performance_report(detailed=True)
            
            # Mettre à jour l'horodatage
            self.last_full_check = current_time
    
    def get_status(self) -> Dict:
        """
        Récupère l'état actuel du moniteur et du modèle
        
        Returns:
            État actuel du moniteur et du modèle
        """
        model_status = self.performance_tracker.get_model_status()
        
        status = {
            "timestamp": datetime.now().isoformat(),
            "model_status": model_status,
            "alerts_enabled": self.enable_alerts,
            "check_interval_hours": self.check_interval,
            "last_full_check": self.last_full_check.isoformat() if self.last_full_check else None,
            "prediction_cache_size": len(self.prediction_cache),
            "validated_predictions": sum(1 for p in self.prediction_cache.values() if p.get("validated", False))
        }
        
        return status
    
    def reset_after_retraining(self, new_model_path: Optional[str] = None) -> None:
        """
        Réinitialise le moniteur après un réentraînement du modèle
        
        Args:
            new_model_path: Chemin vers le nouveau modèle (si différent)
        """
        # Charger le nouveau modèle si fourni
        if new_model_path:
            try:
                self.model = LSTMModel()
                self.model.load(new_model_path)
                logger.info(f"Nouveau modèle chargé pour monitoring: {new_model_path}")
            except Exception as e:
                logger.error(f"Erreur lors du chargement du nouveau modèle: {str(e)}")
        
        # Réinitialiser le tracker de performances
        self.performance_tracker.reset_drift_detection()
        
        # Vider le cache de prédictions
        self.prediction_cache = {}
        
        # Réinitialiser l'horodatage de vérification
        self.last_full_check = datetime.now()
        
        logger.info("Moniteur de modèle réinitialisé après réentraînement")
    
    def generate_monitoring_dashboard(self, output_file: Optional[str] = None) -> str:
        """
        Génère un tableau de bord de monitoring complet
        
        Args:
            output_file: Nom du fichier de sortie
            
        Returns:
            Chemin vers le dashboard généré
        """
        if output_file is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_file = f"model_dashboard_{timestamp}.html"
        
        dashboard_path = os.path.join(self.output_dir, "dashboards", output_file)
        os.makedirs(os.path.dirname(dashboard_path), exist_ok=True)
        
        # Récupérer les données de statut
        status = self.get_status()
        
        # Générer les graphiques de performance
        performance_viz = None
        if len(self.performance_tracker.accuracy_history) >= 10:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            viz_dir = os.path.join(self.output_dir, "visualizations")
            os.makedirs(viz_dir, exist_ok=True)
            
            viz_file = os.path.join(viz_dir, f"dashboard_viz_{timestamp}.png")
            
            # Créer le graphique général
            self._generate_dashboard_visualization(viz_file)
            
            performance_viz = viz_file
        
        # Générer le HTML
        html_content = self._generate_dashboard_html(status, performance_viz)
        
        # Sauvegarder le fichier
        with open(dashboard_path, 'w') as f:
            f.write(html_content)
        
        logger.info(f"Tableau de bord de monitoring généré: {dashboard_path}")
        return dashboard_path
    
    def _generate_dashboard_visualization(self, output_file: str) -> None:
        """
        Génère une visualisation pour le tableau de bord
        
        Args:
            output_file: Chemin du fichier de sortie
        """
        # Créer une figure avec 2x2 sous-graphiques
        fig, axs = plt.subplots(2, 2, figsize=(16, 12))
        
        # Configurer le style
        plt.style.use('ggplot')
        
        # 1. Précision globale au fil du temps
        ax1 = axs[0, 0]
        
        accuracy_values = list(self.performance_tracker.accuracy_history)
        x = range(len(accuracy_values))
        
        ax1.plot(x, accuracy_values, '-', color='blue', linewidth=2)
        
        # Ajouter une ligne de tendance
        if len(accuracy_values) > 5:
            z = np.polyfit(x, accuracy_values, 1)
            p = np.poly1d(z)
            ax1.plot(x, p(x), "r--", alpha=0.7)
        
        # Ajouter des lignes de seuil
        ax1.axhline(y=self.performance_tracker.performance_threshold, color='orange', 
                   linestyle='--', alpha=0.7)
        
        ax1.set_title('Précision Globale', fontsize=14)
        ax1.set_xlabel('Échantillons')
        ax1.set_ylabel('Précision')
        ax1.grid(True, alpha=0.3)
        
        # 2. Précision par horizon
        ax2 = axs[0, 1]
        
        # Pour chaque horizon, calculer la précision moyenne récente
        horizons = []
        accuracies = []
        
        for horizon_key, metrics in self.performance_tracker.horizon_metrics.items():
            if len(metrics["accuracy"]) >= 10:
                # Extraire le numéro d'horizon
                horizon_name = horizon_key.replace("horizon_", "h")
                
                # Calculer la précision moyenne récente
                recent_accuracy = np.mean(list(metrics["accuracy"])[-10:])
                
                horizons.append(horizon_name)
                accuracies.append(recent_accuracy)
        
        if horizons:
            # Tracer les barres de précision
            bar_colors = ['green' if acc >= self.performance_tracker.performance_threshold else 
                        'orange' if acc >= self.performance_tracker.performance_threshold - self.performance_tracker.alert_threshold else 
                        'red' for acc in accuracies]
            
            bars = ax2.bar(horizons, accuracies, color=bar_colors)
            
            # Ajouter les valeurs au-dessus des barres
            for bar, acc in zip(bars, accuracies):
                height = bar.get_height()
                ax2.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                       f'{acc:.2f}', ha='center', va='bottom')
            
            # Ajouter une ligne pour le seuil
            ax2.axhline(y=self.performance_tracker.performance_threshold, color='orange', 
                       linestyle='--', alpha=0.7)
        
        ax2.set_title('Précision par Horizon', fontsize=14)
        ax2.set_ylim([0, 1])
        ax2.set_ylabel('Précision')
        ax2.grid(True, alpha=0.3)
        
        # 3. Matrice de confusion pour l'horizon court terme
        ax3 = axs[1, 0]
        
        # Trouver l'horizon court terme
        short_term_key = None
        for key in self.performance_tracker.horizon_metrics.keys():
            if "12" in key or "short" in key:
                short_term_key = key
                break
        
        if short_term_key and len(self.performance_tracker.horizon_metrics[short_term_key]["direction_tp"]) > 0:
            # Calculer la matrice de confusion
            tp = np.sum(list(self.performance_tracker.horizon_metrics[short_term_key]["direction_tp"])[-20:])
            fp = np.sum(list(self.performance_tracker.horizon_metrics[short_term_key]["direction_fp"])[-20:])
            tn = np.sum(list(self.performance_tracker.horizon_metrics[short_term_key]["direction_tn"])[-20:])
            fn = np.sum(list(self.performance_tracker.horizon_metrics[short_term_key]["direction_fn"])[-20:])
            
            confusion_matrix = np.array([[tn, fp], [fn, tp]])
            
            # Tracer la matrice de confusion
            sns.heatmap(confusion_matrix, annot=True, fmt='d', cmap="Blues", cbar=False,
                      xticklabels=["Négatif", "Positif"], yticklabels=["Négatif", "Positif"], ax=ax3)
            
            ax3.set_title(f'Matrice de Confusion - {short_term_key}', fontsize=14)
            ax3.set_xlabel('Prédiction')
            ax3.set_ylabel('Réalité')
        else:
            ax3.text(0.5, 0.5, "Données insuffisantes", ha='center', va='center', fontsize=14)
            ax3.set_title('Matrice de Confusion', fontsize=14)
        
        # 4. Alertes et état du modèle
        ax4 = axs[1, 1]
        
        # Afficher un résumé de l'état du modèle
        model_status = self.performance_tracker.get_model_status()
        
        # Nettoyer le sous-graphique
        ax4.axis('off')
        
        # Créer une table avec les métriques
        status_text = [
            f"État du modèle: {model_status['model_health'].upper()}",
            f"Réentraînement nécessaire: {'OUI' if model_status['requires_retraining'] else 'Non'}",
            f"Précision récente: {model_status['overall_metrics']['accuracy']:.3f}",
            f"Tendance: {'↗' if model_status['overall_metrics']['accuracy_trend'] > 0 else '↘' if model_status['overall_metrics']['accuracy_trend'] < 0 else '→'}",
            f"Alertes: {len(self.performance_tracker.alert_history)}",
            f"Échantillons: {model_status['samples_count']}"
        ]
        
        # Ajouter des infos par horizon
        horizon_text = ["", "Statut par horizon:"]
        for horizon, metrics in model_status.get("horizon_metrics", {}).items():
            horizon_name = horizon.replace("horizon_", "h")
            horizon_text.append(f"{horizon_name}: {metrics['accuracy']:.3f} - {metrics['health'].upper()}")
        
        # Combiner les textes
        all_text = status_text + horizon_text
        
        # Afficher comme texte
        status_color = {
            "good": "green",
            "degrading": "orange",
            "critical": "red",
            "monitoring": "blue"
        }.get(model_status["model_health"], "black")
        
        # Fond coloré pour l'état
        ax4.text(0.5, 0.95, status_text[0], ha='center', va='top', fontsize=16,
                color='white', fontweight='bold',
                bbox=dict(facecolor=status_color, alpha=0.7, boxstyle='round,pad=0.5'))
        
        # Autres métriques
        y_pos = 0.85
        for line in status_text[1:]:
            ax4.text(0.1, y_pos, line, ha='left', va='top', fontsize=14)
            y_pos -= 0.07
        
        # Infos par horizon
        y_pos -= 0.05
        for line in horizon_text:
            ax4.text(0.1, y_pos, line, ha='left', va='top', fontsize=14)
            y_pos -= 0.07
        
        ax4.set_title('État du Modèle', fontsize=14)
        
        # Configurer la mise en page globale
        plt.tight_layout()
        
        # Sauvegarder la figure
        plt.savefig(output_file)
        plt.close(fig)
    
    def _generate_dashboard_html(self, status: Dict, performance_viz: Optional[str] = None) -> str:
        """
        Génère le HTML pour le tableau de bord
        
        Args:
            status: État actuel du modèle
            performance_viz: Chemin vers la visualisation de performance
            
        Returns:
            Contenu HTML du tableau de bord
        """
        model_status = status["model_status"]
        
        # Déterminer la couleur de statut
        status_color = {
            "good": "#28a745",
            "degrading": "#ffc107",
            "critical": "#dc3545",
            "monitoring": "#17a2b8"
        }.get(model_status["model_health"], "#6c757d")
        
        # Générer le HTML
        html = f"""<!DOCTYPE html>
<html lang="fr">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Tableau de Bord - Monitoring du Modèle LSTM</title>
    <style>
        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            margin: 0;
            padding: 0;
            background-color: #f8f9fa;
            color: #343a40;
        }}
        
        .container {{
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
        }}
        
        header {{
            background-color: #343a40;
            color: white;
            padding: 15px 0;
            text-align: center;
            margin-bottom: 30px;
        }}
        
        .status-card {{
            background-color: white;
            border-radius: 5px;
            border-top: 5px solid {status_color};
            padding: 20px;
            margin-bottom: 20px;
            box-shadow: 0 2px 5px rgba(0,0,0,0.1);
        }}
        
        .metrics-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fill, minmax(280px, 1fr));
            gap: 20px;
            margin-bottom: 30px;
        }}
        
        .metric-card {{
            background-color: white;
            border-radius: 5px;
            padding: 20px;
            box-shadow: 0 2px 5px rgba(0,0,0,0.1);
        }}
        
        .metric-value {{
            font-size: 2rem;
            font-weight: bold;
            margin: 10px 0;
        }}
        
        .visualization {{
            background-color: white;
            border-radius: 5px;
            padding: 20px;
            margin-bottom: 30px;
            box-shadow: 0 2px 5px rgba(0,0,0,0.1);
            text-align: center;
        }}
        
        .visualization img {{
            max-width: 100%;
            height: auto;
        }}
        
        .horizon-table {{
            width: 100%;
            border-collapse: collapse;
            margin-bottom: 30px;
        }}
        
        .horizon-table th, .horizon-table td {{
            padding: 12px 15px;
            text-align: left;
            border-bottom: 1px solid #ddd;
        }}
        
        .horizon-table tr:hover {{
            background-color: #f5f5f5;
        }}
        
        .alert {{
            padding: 15px;
            border-radius: 5px;
            margin-bottom: 20px;
            color: white;
            background-color: {status_color};
        }}
        
        footer {{
            text-align: center;
            padding: 20px 0;
            color: #6c757d;
            font-size: 0.9rem;
            border-top: 1px solid #dee2e6;
            margin-top: 50px;
        }}
    </style>
</head>
<body>
    <header>
        <h1>Monitoring du Modèle LSTM</h1>
    </header>
    
    <div class="container">
        <div class="alert">
            <h2>État du modèle: {model_status["model_health"].upper()}</h2>
            <p>Réentraînement recommandé: {"OUI" if model_status["requires_retraining"] else "Non"}</p>
        </div>
        
        <div class="metrics-grid">
            <div class="metric-card">
                <h3>Précision Globale</h3>
                <div class="metric-value">{model_status["overall_metrics"]["accuracy"]:.2f}</div>
                <p>Tendance: {model_status["overall_metrics"]["accuracy_trend"]:.4f}</p>
            </div>
            
            <div class="metric-card">
                <h3>Perte</h3>
                <div class="metric-value">{model_status["overall_metrics"]["loss"]:.4f}</div>
                <p>Tendance: {model_status["overall_metrics"]["loss_trend"]:.4f}</p>
            </div>
            
            <div class="metric-card">
                <h3>Échantillons</h3>
                <div class="metric-value">{model_status["samples_count"]}</div>
                <p>Alertes: {status["alerts_enabled"]}</p>
            </div>
            
            <div class="metric-card">
                <h3>Dernière vérification</h3>
                <div class="metric-value">{status["last_full_check"] if status["last_full_check"] else "Jamais"}</div>
                <p>Intervalle: {status["check_interval_hours"]} heures</p>
            </div>
        </div>
        
        <div class="status-card">
            <h2>Performance par horizon</h2>
            <table class="horizon-table">
                <thead>
                    <tr>
                        <th>Horizon</th>
                        <th>Précision</th>
                        <th>Tendance</th>
                        <th>Statut</th>
                    </tr>
                </thead>
                <tbody>"""
        
        # Ajouter les lignes pour chaque horizon
        for horizon_key, horizon_metrics in model_status.get("horizon_metrics", {}).items():
            # Déterminer la couleur en fonction du statut
            horizon_color = {
                "good": "#28a745",
                "degrading": "#ffc107",
                "critical": "#dc3545"
            }.get(horizon_metrics["health"], "#6c757d")
            
            horizon_name = horizon_key.replace("horizon_", "h")
            
            html += f"""
                    <tr>
                        <td>{horizon_name}</td>
                        <td>{horizon_metrics["accuracy"]:.3f}</td>
                        <td>{horizon_metrics["trend"]:.4f}</td>
                        <td style="color: {horizon_color}; font-weight: bold;">{horizon_metrics["health"].upper()}</td>
                    </tr>"""
        
        html += """
                </tbody>
            </table>
        </div>
        """
        
        # Ajouter la visualisation si disponible
        if performance_viz:
            # Convertir le chemin absolu en chemin relatif si possible
            viz_filename = os.path.basename(performance_viz)
            
            html += f"""
        <div class="visualization">
            <h2>Visualisation des performances</h2>
            <img src="{viz_filename}" alt="Performance du modèle" />
        </div>
            """
        
        # Ajouter l'historique des alertes récentes
        if self.performance_tracker.alert_history:
            html += """
        <div class="status-card">
            <h2>Alertes récentes</h2>
            <table class="horizon-table">
                <thead>
                    <tr>
                        <th>Date</th>
                        <th>Type</th>
                        <th>Détails</th>
                    </tr>
                </thead>
                <tbody>"""
            
            # Ajouter les 5 dernières alertes
            for alert in self.performance_tracker.alert_history[-5:]:
                alert_timestamp = alert.get("timestamp", "")
                alert_type = "Drift" if alert.get("drift_detected") else "Performance"
                drifting_horizons = ", ".join(alert.get("drifting_horizons", []))
                if not drifting_horizons and alert.get("drift_detected"):
                    drifting_horizons = "Multiple"
                
                details = f"Modèle: {alert.get('model_health', '').upper()}"
                if drifting_horizons:
                    details += f", Horizons: {drifting_horizons}"
                
                html += f"""
                    <tr>
                        <td>{alert_timestamp}</td>
                        <td>{alert_type}</td>
                        <td>{details}</td>
                    </tr>"""
            
            html += """
                </tbody>
            </table>
        </div>
            """
        
        # Finaliser le HTML
        html += f"""
        <footer>
            <p>Généré le {datetime.now().strftime('%Y-%m-%d à %H:%M:%S')}</p>
        </footer>
    </div>
</body>
</html>
        """
        
        return html


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Surveillance des performances du modèle LSTM")
    parser.add_argument("--model", type=str, help="Chemin vers le modèle LSTM")
    parser.add_argument("--dashboard", action="store_true", help="Générer un tableau de bord")
    parser.add_argument("--report", action="store_true", help="Générer un rapport de performance")
    
    args = parser.parse_args()
    
    # Initialiser le moniteur
    monitor = ModelMonitor(model_path=args.model)
    
    # Générer un tableau de bord si demandé
    if args.dashboard:
        dashboard_path = monitor.generate_monitoring_dashboard()
        print(f"Tableau de bord généré: {dashboard_path}")
    
    # Générer un rapport si demandé
    if args.report:
        report_path = monitor.performance_tracker.generate_performance_report()
        if report_path:
            print(f"Rapport de performance généré: {report_path}")
        else:
            print("Impossible de générer le rapport (données insuffisantes)")
    
    # Afficher l'état actuel si aucune option spécifiée
    if not args.dashboard and not args.report:
        status = monitor.get_status()
        print("État du modèle:")
        print(f"  Santé: {status['model_status']['model_health'].upper()}")
        print(f"  Réentraînement nécessaire: {'OUI' if status['model_status']['requires_retraining'] else 'Non'}")
        
        if status['model_status']['overall_metrics']['accuracy'] is not None:
            print(f"  Précision récente: {status['model_status']['overall_metrics']['accuracy']:.3f}")
        else:
            print("  Données insuffisantes pour calculer la précision")