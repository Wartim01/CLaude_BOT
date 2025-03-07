"""
Orchestrateur de prédictions qui gère l'ensemble des modèles d'IA
et fournit une interface unifiée pour les prédictions de prix et tendances
"""
import os
import json
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Union, Any
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor
import time
import traceback

from config.config import DATA_DIR, MODEL_CHECKPOINTS_DIR
from utils.logger import setup_logger
from ai.models.lstm_model import LSTMModel
from ai.models.feature_engineering import FeatureEngineering
from ai.models.continuous_learning import AdvancedContinuousLearning
from ai.models.model_validator import ModelValidator

logger = setup_logger("prediction_orchestrator")

class PredictionOrchestrator:
    """
    Gère l'ensemble des modèles d'IA et fournit une interface unifiée pour les prédictions
    """
    def __init__(self, models_dir: Optional[str] = None):
        """
        Initialise l'orchestrateur avec les répertoires par défaut
        
        Args:
            models_dir: Répertoire des modèles (utilise la valeur par défaut si None)
        """
        self.models_dir = models_dir or os.path.join(DATA_DIR, "models", "production")
        
        # Créer le répertoire si nécessaire
        os.makedirs(self.models_dir, exist_ok=True)
        
        # Ingénierie des caractéristiques
        self.feature_engineering = FeatureEngineering(save_scalers=True)
        
        # Dictionnaire des modèles chargés
        self.models = {}
        
        # Configuration par défaut
        self.config = {
            "default_model": "lstm",
            "ensemble_enabled": True,
            "continuous_learning_enabled": True,
            "prediction_cache_duration": 300,  # secondes
            "confidence_threshold": 0.65,
            "fallback_to_consensus": True
        }
        
        # Cache de prédictions (pour éviter de recalculer)
        self.prediction_cache = {}
        
        # Statistiques de prédiction
        self.prediction_stats = {
            "total_predictions": 0,
            "successful_predictions": 0,
            "failed_predictions": 0,
            "ensemble_disagreements": 0,
            "cache_hits": 0,
            "model_usage": {}
        }
        
        # Système d'apprentissage continu
        self.continuous_learning = {}
        
        # Chargement des modèles par défaut
        self._load_default_models()
        
        # Chargement de la configuration
        self._load_config()
    
    def _load_default_models(self) -> None:
        """Charge les modèles par défaut pour chaque paire de trading"""
        try:
            # Rechercher tous les modèles disponibles dans le répertoire
            model_files = [f for f in os.listdir(self.models_dir) if f.endswith(('.h5', '.keras'))]
            
            # Modèles par défaut à charger immédiatement
            default_models = ["lstm_final.keras", "enhanced_lstm_model.h5"]
            
            # Charger les modèles par défaut
            for model_file in model_files:
                if model_file in default_models:
                    model_path = os.path.join(self.models_dir, model_file)
                    model_type = "lstm" if "lstm" in model_file.lower() else "unknown"
                    
                    self._load_model(model_path, model_type)
            
            logger.info(f"Modèles chargés: {list(self.models.keys())}")
        except Exception as e:
            logger.error(f"Erreur lors du chargement des modèles par défaut: {str(e)}")
    
    def _load_model(self, model_path: str, model_type: str) -> bool:
        """
        Charge un modèle depuis un fichier
        
        Args:
            model_path: Chemin du modèle
            model_type: Type de modèle ('lstm', 'transformer', etc.)
            
        Returns:
            Succès du chargement
        """
        try:
            if model_type.lower() == "lstm":
                model = LSTMModel()
                model.load(model_path)
                
                # Extraire un identifiant du nom du fichier
                model_id = os.path.basename(model_path).replace('.keras', '').replace('.h5', '')
                
                # Stocker le modèle
                self.models[model_id] = {
                    "model": model,
                    "type": model_type,
                    "path": model_path,
                    "loaded_at": datetime.now().isoformat(),
                    "prediction_count": 0,
                    "avg_prediction_time": 0
                }
                
                # Créer un système d'apprentissage continu pour ce modèle
                if self.config["continuous_learning_enabled"]:
                    self.continuous_learning[model_id] = AdvancedContinuousLearning(
                        model=model,
                        feature_engineering=self.feature_engineering,
                        replay_memory_size=5000,
                        drift_threshold=0.1
                    )
                
                logger.info(f"Modèle '{model_id}' chargé: {model_path}")
                return True
            else:
                logger.warning(f"Type de modèle non supporté: {model_type}")
                return False
        except Exception as e:
            logger.error(f"Erreur lors du chargement du modèle: {str(e)}")
            logger.error(traceback.format_exc())
            return False
    
    def _load_config(self) -> None:
        """Charge la configuration depuis le fichier"""
        config_path = os.path.join(self.models_dir, "prediction_config.json")
        
        if os.path.exists(config_path):
            try:
                with open(config_path, 'r') as f:
                    config = json.load(f)
                
                # Mettre à jour la config avec les valeurs chargées
                self.config.update(config)
                
                logger.info(f"Configuration chargée: {self.config}")
            except Exception as e:
                logger.error(f"Erreur lors du chargement de la configuration: {str(e)}")
    
    def _save_config(self) -> None:
        """Sauvegarde la configuration dans un fichier"""
        config_path = os.path.join(self.models_dir, "prediction_config.json")
        
        try:
            with open(config_path, 'w') as f:
                json.dump(self.config, f, indent=2)
        except Exception as e:
            logger.error(f"Erreur lors de la sauvegarde de la configuration: {str(e)}")
    
    def get_prediction(self, symbol: str, timeframe: str, data: pd.DataFrame, 
                    model_id: Optional[str] = None, force_recalculation: bool = False) -> Dict:
        """
        Obtient une prédiction pour un symbole et une période spécifiques
        
        Args:
            symbol: Symbole de la paire de trading
            timeframe: Période de temps ('15m', '1h', '4h', etc.)
            data: DataFrame avec les données OHLCV récentes
            model_id: ID du modèle à utiliser (utilise la valeur par défaut si None)
            force_recalculation: Force le recalcul de la prédiction (ignore le cache)
            
        Returns:
            Prédiction avec confiance et métadonnées
        """
        # Incrémenter le compteur de prédictions
        self.prediction_stats["total_predictions"] += 1
        
        # Créer une clé de cache unique
        cache_key = f"{symbol}_{timeframe}_{data.index[-1].isoformat()}"
        
        # Vérifier si la prédiction est dans le cache
        if not force_recalculation and cache_key in self.prediction_cache:
            cache_entry = self.prediction_cache[cache_key]
            
            # Vérifier si le cache est encore valide
            cache_age = (datetime.now() - cache_entry["timestamp"]).total_seconds()
            
            if cache_age < self.config["prediction_cache_duration"]:
                # Utiliser la prédiction en cache
                self.prediction_stats["cache_hits"] += 1
                return cache_entry["prediction"]
        
        # Déterminer le modèle à utiliser
        selected_model_id = model_id or self.config["default_model"]
        
        # Vérifier si le modèle est disponible
        if selected_model_id not in self.models:
            logger.warning(f"Modèle '{selected_model_id}' non disponible. Utilisation du consensus.")
            
            # Utiliser le consensus de tous les modèles disponibles
            if self.config["fallback_to_consensus"] and self.models:
                return self._get_ensemble_prediction(symbol, timeframe, data)
            else:
                self.prediction_stats["failed_predictions"] += 1
                return self._generate_error_prediction(
                    symbol=symbol,
                    timeframe=timeframe,
                    error=f"Modèle '{selected_model_id}' non disponible"
                )
        
        # Obtenir le modèle
        model_data = self.models[selected_model_id]
        model = model_data["model"]
        
        try:
            # Mesurer le temps de prédiction
            start_time = time.time()
            
            # Prétraiter les données
            featured_data = self.feature_engineering.create_features(
                data, 
                include_time_features=True,
                include_price_patterns=True
            )
            
            # Normaliser les caractéristiques
            normalized_data = self.feature_engineering.scale_features(
                featured_data,
                is_training=False,
                method='standard',
                feature_group='lstm'
            )
            
            # Création des séquences d'entrée
            # La méthode spécifique dépend de l'interface de chaque modèle
            if hasattr(model, 'predict'):
                # Pour des modèles avec une méthode predict personnalisée
                prediction = model.predict(normalized_data)
            else:
                # Pour les modèles standards TensorFlow/Keras
                # Créer les séquences d'entrée
                input_length = getattr(model, 'input_length', 60)
                
                if hasattr(model, 'prediction_horizons'):
                    horizons = model.prediction_horizons
                else:
                    # Valeurs par défaut
                    horizons = [12, 24, 96]
                
                X = self.feature_engineering.prepare_lstm_data(
                    normalized_data,
                    sequence_length=input_length,
                    prediction_horizon=horizons[0],  # Utiliser l'horizon le plus court
                    is_training=False
                )
                
                # Faire la prédiction
                raw_predictions = model.model.predict(X)
                
                # Traiter les prédictions
                prediction = self._process_raw_predictions(
                    raw_predictions=raw_predictions,
                    symbol=symbol,
                    timeframe=timeframe,
                    horizons=horizons
                )
            
            # Calculer le temps d'exécution
            execution_time = time.time() - start_time
            
            # Mise à jour des statistiques du modèle
            model_data["prediction_count"] += 1
            
            # Mise à jour du temps moyen de prédiction (moyenne mobile)
            prev_avg = model_data["avg_prediction_time"]
            count = model_data["prediction_count"]
            model_data["avg_prediction_time"] = (prev_avg * (count - 1) + execution_time) / count
            
            # Mettre en cache la prédiction
            self.prediction_cache[cache_key] = {
                "prediction": prediction,
                "timestamp": datetime.now()
            }
            
            # Mise à jour des statistiques d'utilisation du modèle
            if selected_model_id not in self.prediction_stats["model_usage"]:
                self.prediction_stats["model_usage"][selected_model_id] = 0
            self.prediction_stats["model_usage"][selected_model_id] += 1
            
            # Mise à jour des statistiques de prédiction
            self.prediction_stats["successful_predictions"] += 1
            
            # Envoi des données pour l'apprentissage continu
            if self.config["continuous_learning_enabled"] and selected_model_id in self.continuous_learning:
                # Cette opération est asynchrone pour ne pas bloquer les prédictions
                self._schedule_continuous_learning(selected_model_id, data)
            
            return prediction
        
        except Exception as e:
            logger.error(f"Erreur lors de la prédiction avec le modèle '{selected_model_id}': {str(e)}")
            logger.error(traceback.format_exc())
            
            # Mise à jour des statistiques de prédiction
            self.prediction_stats["failed_predictions"] += 1
            
            # Utiliser le consensus en cas d'erreur si activé
            if self.config["fallback_to_consensus"] and len(self.models) > 1:
                logger.info(f"Utilisation du consensus suite à une erreur du modèle '{selected_model_id}'")
                return self._get_ensemble_prediction(symbol, timeframe, data, exclude_model=selected_model_id)
            else:
                return self._generate_error_prediction(
                    symbol=symbol,
                    timeframe=timeframe,
                    error=str(e)
                )
    
    def _process_raw_predictions(self, raw_predictions: List[np.ndarray], 
                              symbol: str, timeframe: str, horizons: List[int]) -> Dict:
        """
        Traite les prédictions brutes du modèle
        
        Args:
            raw_predictions: Liste des prédictions brutes du modèle
            symbol: Symbole de la paire de trading
            timeframe: Période de temps
            horizons: Liste des horizons de prédiction (en périodes)
            
        Returns:
            Prédiction structurée
        """
        prediction = {
            "symbol": symbol,
            "timeframe": timeframe,
            "generated_at": datetime.now().isoformat(),
            "horizons": {}
        }
        
        # Convertir les horizons en périodes en descriptions lisibles
        horizon_names = {}
        for horizon in horizons:
            if timeframe == '15m':
                hours = horizon * 15 / 60
                horizon_names[horizon] = f"{int(hours)}h" if hours.is_integer() else f"{hours}h"
            elif timeframe == '1h':
                horizon_names[horizon] = f"{horizon}h"
            elif timeframe == '4h':
                horizon_names[horizon] = f"{horizon * 4}h"
            elif timeframe == '1d':
                horizon_names[horizon] = f"{horizon}d"
            else:
                horizon_names[horizon] = f"{horizon} périodes"
        
        # Traiter chaque horizon de prédiction
        for i, horizon in enumerate(horizons):
            # En supposant que raw_predictions soit une liste de tableaux numpy
            # où chaque tableau correspond à un horizon
            if i < len(raw_predictions):
                # Prendre la dernière prédiction (la plus récente)
                if len(raw_predictions[i]) > 0:
                    prob_up = float(raw_predictions[i][-1][0])
                    
                    # Dériver la direction et la confiance
                    if prob_up > 0.5:
                        direction = "UP"
                        confidence = (prob_up - 0.5) * 2  # Transformer [0.5, 1] en [0, 1]
                    else:
                        direction = "DOWN"
                        confidence = (0.5 - prob_up) * 2  # Transformer [0, 0.5] en [0, 1]
                    
                    # Seuil de confiance pour les signaux
                    if confidence < 0.2:
                        signal_type = "NEUTRAL"
                    elif confidence < 0.4:
                        signal_type = "WEAK_" + direction
                    elif confidence < 0.7:
                        signal_type = "MODERATE_" + direction
                    else:
                        signal_type = "STRONG_" + direction
                    
                    # Stocker la prédiction pour cet horizon
                    prediction["horizons"][horizon_names[horizon]] = {
                        "raw_value": prob_up,
                        "direction": direction,
                        "confidence": float(confidence),
                        "signal_type": signal_type
                    }
        
        # Ajouter la prédiction principale (horizon à court terme)
        if horizons and horizon_names[horizons[0]] in prediction["horizons"]:
            prediction["primary"] = prediction["horizons"][horizon_names[horizons[0]]]
        else:
            prediction["primary"] = {
                "direction": "NEUTRAL",
                "confidence": 0.0,
                "signal_type": "NEUTRAL"
            }
        
        return prediction
    
    def _get_ensemble_prediction(self, symbol: str, timeframe: str, data: pd.DataFrame,
                              exclude_model: Optional[str] = None) -> Dict:
        """
        Obtient une prédiction consensuelle à partir de tous les modèles disponibles
        
        Args:
            symbol: Symbole de la paire de trading
            timeframe: Période de temps
            data: DataFrame avec les données OHLCV
            exclude_model: ID du modèle à exclure de l'ensemble (optionnel)
            
        Returns:
            Prédiction consensuelle
        """
        # Liste de tous les modèles disponibles sauf celui exclu
        available_models = [
            model_id for model_id in self.models.keys()
            if model_id != exclude_model
        ]
        
        if not available_models:
            return self._generate_error_prediction(
                symbol=symbol,
                timeframe=timeframe,
                error="Aucun modèle disponible pour l'ensemble"
            )
        
        # Obtenir les prédictions de chaque modèle
        model_predictions = {}
        
        for model_id in available_models:
            try:
                # Forcer le recalcul pour chaque modèle
                prediction = self.get_prediction(
                    symbol=symbol,
                    timeframe=timeframe,
                    data=data,
                    model_id=model_id,
                    force_recalculation=True
                )
                
                model_predictions[model_id] = prediction
            except Exception as e:
                logger.error(f"Erreur avec le modèle '{model_id}' pour l'ensemble: {str(e)}")
        
        if not model_predictions:
            return self._generate_error_prediction(
                symbol=symbol,
                timeframe=timeframe,
                error="Toutes les prédictions de l'ensemble ont échoué"
            )
        
        # Créer la prédiction consensuelle
        ensemble_prediction = {
            "symbol": symbol,
            "timeframe": timeframe,
            "generated_at": datetime.now().isoformat(),
            "type": "ensemble",
            "contributing_models": list(model_predictions.keys()),
            "horizons": {}
        }
        
        # Fusionner les prédictions pour chaque horizon
        all_horizons = set()
        for model_id, prediction in model_predictions.items():
            all_horizons.update(prediction.get("horizons", {}).keys())
        
        # Pour chaque horizon, calculer la moyenne des prédictions
        for horizon in all_horizons:
            valid_predictions = []
            
            for model_id, prediction in model_predictions.items():
                if horizon in prediction.get("horizons", {}):
                    valid_predictions.append(prediction["horizons"][horizon])
            
            if valid_predictions:
                # Moyenne des valeurs brutes
                raw_values = [p.get("raw_value", 0.5) for p in valid_predictions if "raw_value" in p]
                
                if raw_values:
                    avg_raw = sum(raw_values) / len(raw_values)
                    
                    # Dériver la direction et la confiance
                    if avg_raw > 0.5:
                        direction = "UP"
                        confidence = (avg_raw - 0.5) * 2
                    else:
                        direction = "DOWN"
                        confidence = (0.5 - avg_raw) * 2
                    
                    # Seuil de confiance pour les signaux
                    if confidence < 0.2:
                        signal_type = "NEUTRAL"
                    elif confidence < 0.4:
                        signal_type = "WEAK_" + direction
                    elif confidence < 0.7:
                        signal_type = "MODERATE_" + direction
                    else:
                        signal_type = "STRONG_" + direction
                    
                    # Vérifier le désaccord entre modèles
                    directions = [p.get("direction", "NEUTRAL") for p in valid_predictions]
                    unique_directions = set(directions)
                    
                    # S'il y a un désaccord, réduire la confiance
                    if len(unique_directions) > 1 and "NEUTRAL" not in unique_directions:
                        confidence *= 0.8
                        self.prediction_stats["ensemble_disagreements"] += 1
                    
                    # Stocker la prédiction pour cet horizon
                    ensemble_prediction["horizons"][horizon] = {
                        "raw_value": float(avg_raw),
                        "direction": direction,
                        "confidence": float(confidence),
                        "signal_type": signal_type,
                        "agreement": (len(directions) - len(unique_directions) + 1) / len(directions)
                    }
        
        # Ajouter la prédiction principale (horizon à court terme)
        primary_horizon = min(ensemble_prediction["horizons"].keys()) if ensemble_prediction["horizons"] else None
        
        if primary_horizon:
            ensemble_prediction["primary"] = ensemble_prediction["horizons"][primary_horizon]
        else:
            ensemble_prediction["primary"] = {
                "direction": "NEUTRAL",
                "confidence": 0.0,
                "signal_type": "NEUTRAL"
            }
        
        return ensemble_prediction
    
    def _generate_error_prediction(self, symbol: str, timeframe: str, error: str) -> Dict:
        """
        Génère une prédiction d'erreur avec des valeurs neutres
        
        Args:
            symbol: Symbole de la paire de trading
            timeframe: Période de temps
            error: Description de l'erreur
            
        Returns:
            Prédiction d'erreur
        """
        return {
            "symbol": symbol,
            "timeframe": timeframe,
            "generated_at": datetime.now().isoformat(),
            "error": error,
            "horizons": {},
            "primary": {
                "direction": "NEUTRAL",
                "confidence": 0.0,
                "signal_type": "NEUTRAL"
            }
        }
    
    def _schedule_continuous_learning(self, model_id: str, data: pd.DataFrame) -> None:
        """
        Planifie l'apprentissage continu de manière asynchrone
        
        Args:
            model_id: ID du modèle à mettre à jour
            data: Données récentes pour l'apprentissage
        """
        if model_id not in self.continuous_learning:
            return
        
        # Utiliser un thread séparé pour ne pas bloquer les prédictions
        def continuous_learning_task():
            try:
                cl_system = self.continuous_learning[model_id]
                cl_system.process_new_data(data, min_samples=30)
                logger.info(f"Apprentissage continu effectué pour le modèle '{model_id}'")
            except Exception as e:
                logger.error(f"Erreur lors de l'apprentissage continu pour '{model_id}': {str(e)}")
        
        # Exécuter la tâche en arrière-plan
        executor = ThreadPoolExecutor(max_workers=1)
        executor.submit(continuous_learning_task)
    
    def get_available_models(self) -> List[Dict]:
        """
        Récupère la liste des modèles disponibles
        
        Returns:
            Liste des modèles avec leurs métadonnées
        """
        models_info = []
        
        for model_id, model_data in self.models.items():
            models_info.append({
                "id": model_id,
                "type": model_data["type"],
                "path": model_data["path"],
                "loaded_at": model_data["loaded_at"],
                "prediction_count": model_data["prediction_count"],
                "avg_prediction_time": model_data["avg_prediction_time"]
            })
        
        return models_info
    
    def update_config(self, new_config: Dict) -> Dict:
        """
        Met à jour la configuration
        
        Args:
            new_config: Nouvelle configuration
            
        Returns:
            Configuration mise à jour
        """
        # Mettre à jour seulement les champs fournis
        for key, value in new_config.items():
            if key in self.config:
                self.config[key] = value
        
        # Sauvegarder la configuration
        self._save_config()
        
        return self.config
    
    def get_stats(self) -> Dict:
        """
        Récupère les statistiques de prédiction
        
        Returns:
            Statistiques de prédiction
        """
        # Calculer des métriques supplémentaires
        stats = self.prediction_stats.copy()
        
        if stats["total_predictions"] > 0:
            stats["success_rate"] = stats["successful_predictions"] / stats["total_predictions"]
            stats["cache_hit_rate"] = stats["cache_hits"] / stats["total_predictions"]
        else:
            stats["success_rate"] = 0
            stats["cache_hit_rate"] = 0
        
        # Ajouter des informations sur le cache
        stats["cache_size"] = len(self.prediction_cache)
        
        # Ajouter des informations sur l'apprentissage continu
        cl_stats = {}
        for model_id, cl_system in self.continuous_learning.items():
            cl_stats[model_id] = cl_system.get_status()
        
        stats["continuous_learning"] = cl_stats
        
        return stats
    
    def clear_cache(self) -> int:
        """
        Efface le cache de prédictions
        
        Returns:
            Nombre d'entrées effacées
        """
        cache_size = len(self.prediction_cache)
        self.prediction_cache = {}
        return cache_size
    
    def load_additional_model(self, model_path: str, model_type: str) -> bool:
        """
        Charge un modèle supplémentaire
        
        Args:
            model_path: Chemin du modèle
            model_type: Type de modèle
            
        Returns:
            Succès du chargement
        """
        return self._load_model(model_path, model_type)
    
    def unload_model(self, model_id: str) -> bool:
        """
        Décharge un modèle
        
        Args:
            model_id: ID du modèle à décharger
            
        Returns:
            Succès du déchargement
        """
        if model_id in self.models:
            # Supprimer le modèle du dictionnaire
            del self.models[model_id]
            
            # Supprimer le système d'apprentissage continu associé
            if model_id in self.continuous_learning:
                del self.continuous_learning[model_id]
            
            logger.info(f"Modèle '{model_id}' déchargé")
            return True
        
        return False
    
    def evaluate_models(self, data: pd.DataFrame, 
                      symbols: Optional[List[str]] = None,
                      timeframe: str = '1h') -> Dict:
        """
        Évalue les performances des modèles sur un ensemble de données
        
        Args:
            data: DataFrame avec les données OHLCV
            symbols: Liste des symboles à évaluer (utilise un symbole par défaut si None)
            timeframe: Période de temps
            
        Returns:
            Résultats de l'évaluation
        """
        if not symbols:
            symbols = ["BTCUSDT"]  # Valeur par défaut
        
        # Initialiser le validateur
        validator = ModelValidator(feature_engineering=self.feature_engineering)
        
        # Résultats d'évaluation
        results = {}
        
        # Pour chaque modèle
        for model_id, model_data in self.models.items():
            model_results = {}
            
            # Assigner le modèle au validateur
            validator.model = model_data["model"]
            
            # Pour chaque symbole
            for symbol in symbols:
                try:
                    # Évaluer le modèle sur ces données
                    evaluation = validator.evaluate_on_test_set(data)
                    
                    # Stocker les résultats
                    model_results[symbol] = {
                        "avg_accuracy": evaluation["avg_accuracy"],
                        "horizons": {
                            k: {
                                "accuracy": v["direction"]["accuracy"],
                                "precision": v["direction"].get("precision", 0),
                                "recall": v["direction"].get("recall", 0),
                                "f1_score": v["direction"].get("f1_score", 0)
                            }
                            for k, v in evaluation["horizons"].items()
                        }
                    }
                except Exception as e:
                    logger.error(f"Erreur lors de l'évaluation du modèle '{model_id}' sur {symbol}: {str(e)}")
                    model_results[symbol] = {"error": str(e)}
            
            # Calculer la moyenne sur tous les symboles pour ce modèle
            avg_accuracy = np.mean([
                model_results[symbol]["avg_accuracy"]
                for symbol in model_results
                if "avg_accuracy" in model_results[symbol]
            ])
            
            results[model_id] = {
                "by_symbol": model_results,
                "avg_accuracy": float(avg_accuracy) if not np.isnan(avg_accuracy) else 0.0,
                "model_type": model_data["type"],
                "evaluation_time": datetime.now().isoformat()
            }
        
        return {
            "results": results,
            "best_model": max(results.keys(), key=lambda x: results[x]["avg_accuracy"]) if results else None,
            "timestamp": datetime.now().isoformat(),
            "evaluation_data": {
                "timeframe": timeframe,
                "symbols": symbols,
                "data_size": len(data)
            }
        }
    
    def visualize_predictions(self, data: pd.DataFrame, symbol: str, timeframe: str, 
                            model_id: Optional[str] = None) -> Dict:
        """
        Génère des visualisations des prédictions pour analyse
        
        Args:
            data: DataFrame avec les données OHLCV
            symbol: Symbole de la paire de trading
            timeframe: Période de temps
            model_id: ID du modèle à utiliser (utilise la valeur par défaut si None)
            
        Returns:
            Résultats des visualisations avec chemins des fichiers
        """
        # Déterminer le modèle à utiliser
        selected_model_id = model_id or self.config["default_model"]
        
        if selected_model_id not in self.models:
            return {
                "success": False,
                "error": f"Modèle '{selected_model_id}' non disponible"
            }
        
        model_data = self.models[selected_model_id]
        model = model_data["model"]
        
        # Créer un validateur pour les visualisations
        validator = ModelValidator(model, feature_engineering=self.feature_engineering)
        
        try:
            # Prétraiter les données
            X, _, normalized_data = validator.preprocess_data(data)
            
            # Créer le répertoire de visualisations
            viz_dir = os.path.join(DATA_DIR, "predictions", "visualizations")
            os.makedirs(viz_dir, exist_ok=True)
            
            # Générer une visualisation pour chaque horizon
            viz_paths = []
            
            # Déterminer combien d'horizons sont disponibles
            horizons = 3  # Par défaut
            
            if hasattr(model, 'prediction_horizons'):
                horizons = len(model.prediction_horizons)
            
            # Générer les visualisations
            for i in range(horizons):
                # Générer un nom de fichier unique
                filename = f"{symbol}_{timeframe}_{selected_model_id}_h{i}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
                path = os.path.join(viz_dir, filename)
                
                # Créer la visualisation
                validator.visualize_predictions(
                    data=data,
                    horizon_idx=i,
                    start_idx=max(0, len(data) - 500),  # Derniers 500 points
                    end_idx=None
                )
                
                viz_paths.append({
                    "horizon_idx": i,
                    "path": path,
                    "url": f"/predictions/visualizations/{filename}"
                })
            
            return {
                "success": True,
                "model_id": selected_model_id,
                "symbol": symbol,
                "timeframe": timeframe,
                "visualizations": viz_paths,
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Erreur lors de la génération des visualisations: {str(e)}")
            logger.error(traceback.format_exc())
            
            return {
                "success": False,
                "error": str(e)
            }
    
    def compare_models(self, data: pd.DataFrame, symbol: str, timeframe: str) -> Dict:
        """
        Compare les performances de différents modèles sur un même ensemble de données
        
        Args:
            data: DataFrame avec les données OHLCV
            symbol: Symbole de la paire de trading
            timeframe: Période de temps
            
        Returns:
            Comparaison des modèles
        """
        if len(self.models) < 2:
            return {
                "success": False,
                "error": "Au moins deux modèles sont nécessaires pour une comparaison"
            }
        
        try:
            # Obtenir des prédictions de chaque modèle
            model_predictions = {}
            model_metrics = {}
            
            for model_id, model_data in self.models.items():
                # Obtenir les prédictions
                prediction = self.get_prediction(
                    symbol=symbol,
                    timeframe=timeframe,
                    data=data,
                    model_id=model_id,
                    force_recalculation=True
                )
                
                model_predictions[model_id] = prediction
                
                # Calculer des métriques
                validator = ModelValidator(model_data["model"], self.feature_engineering)
                evaluation = validator.evaluate_on_test_set(data)
                
                model_metrics[model_id] = {
                    "avg_accuracy": evaluation.get("avg_accuracy", 0.0),
                    "horizons": evaluation.get("horizons", {})
                }
            
            # Créer une comparaison
            # Trouver le meilleur modèle basé sur l'accuracy moyenne
            best_model = max(model_metrics.keys(), key=lambda x: model_metrics[x]["avg_accuracy"])
            
            # Différences entre les modèles
            differences = {}
            for model_id, metrics in model_metrics.items():
                if model_id != best_model:
                    accuracy_diff = model_metrics[best_model]["avg_accuracy"] - metrics["avg_accuracy"]
                    differences[model_id] = {
                        "accuracy_difference": float(accuracy_diff),
                        "percent_difference": float(accuracy_diff / max(0.0001, metrics["avg_accuracy"]) * 100)
                    }
            
            return {
                "success": True,
                "symbol": symbol,
                "timeframe": timeframe,
                "best_model": best_model,
                "model_metrics": model_metrics,
                "differences": differences,
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Erreur lors de la comparaison des modèles: {str(e)}")
            
            return {
                "success": False,
                "error": str(e)
            }
