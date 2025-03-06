# ai/models/model_validator.py
"""
Module de validation et d'évaluation des performances du modèle LSTM
"""
import os
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Union, Optional
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import json

from ai.models.lstm_model import LSTMModel
from ai.models.feature_engineering import FeatureEngineering
from config.config import DATA_DIR
from utils.logger import setup_logger
from strategies.technical_bounce import TechnicalBounceStrategy
from strategies.market_state import MarketStateAnalyzer
from ai.scoring_engine import ScoringEngine

logger = setup_logger("model_validator")

class ModelValidator:
    """
    Classe pour valider et évaluer les performances du modèle LSTM
    et les comparer aux stratégies de base
    """
    def __init__(self, model: Optional[LSTMModel] = None, 
               feature_engineering: Optional[FeatureEngineering] = None):
        """
        Initialise le ModelValidator
        
        Args:
            model: Instance du modèle LSTM à valider
            feature_engineering: Instance du module d'ingénierie des caractéristiques
        """
        self.model = model
        self.feature_engineering = feature_engineering or FeatureEngineering()
        
        # Pour la comparaison avec la stratégie de base
        self.scoring_engine = ScoringEngine()
        
        # Répertoires pour les résultats
        self.output_dir = os.path.join(DATA_DIR, "models", "validation")
        os.makedirs(self.output_dir, exist_ok=True)
    
    def load_model(self, model_path: str) -> None:
        """
        Charge un modèle sauvegardé
        
        Args:
            model_path: Chemin vers le modèle sauvegardé
        """
        # Si aucun modèle n'est fourni, en créer un nouveau
        if self.model is None:
            self.model = LSTMModel()
        
        # Charger le modèle
        self.model.load(model_path)
        logger.info(f"Modèle chargé: {model_path}")
    
    def prepare_data(self, data: pd.DataFrame) -> Tuple:
        from ai.models.feature_engineering import FeatureEngineering
        fe = FeatureEngineering()
        featured_data = fe.create_features(data, include_time_features=True, include_price_patterns=True)
        normalized_data = fe.scale_features(featured_data, is_training=False, method='standard', feature_group='lstm')
        return featured_data, normalized_data
    
    def evaluate_on_test_set(self, test_data: pd.DataFrame) -> Dict:
        """
        Évalue le modèle sur un ensemble de test
        
        Args:
            test_data: DataFrame avec les données de test
            
        Returns:
            Dictionnaire avec les métriques d'évaluation
        """
        if self.model is None:
            raise ValueError("Le modèle n'a pas été initialisé ou chargé.")
        
        # Préparer les données de test
        featured_data, normalized_data = self.prepare_data(test_data)
        
        # Créer des séquences pour chaque horizon de prédiction
        X_test, y_test = self.feature_engineering.create_multi_horizon_data(
            normalized_data,
            sequence_length=self.model.input_length,
            horizons=self.model.prediction_horizons,
            is_training=True
        )
        
        # Évaluer le modèle
        evaluation = self.model.model.evaluate(X_test, y_test, verbose=1)
        loss_value = evaluation[0] if isinstance(evaluation, (list, tuple)) else evaluation
        
        # Faire des prédictions
        predictions = self.model.model.predict(X_test)
        # Si le modèle retourne un seul tableau, enveloppez-le dans une liste et conservez uniquement le premier target
        if not isinstance(predictions, list):
            predictions = [predictions]
            y_test = [y_test[0]]
        
        # Calculer des métriques détaillées pour chaque horizon
        results = {
            "loss": loss_value,
            "horizons": {}
        }
        
        # Pour chaque horizon disponible dans les predictions
        for h_idx, horizon in enumerate(self.model.prediction_horizons):
            if h_idx >= len(predictions):
                break  # Stop if le modèle ne fournit pas cette sortie
            horizon_key = f"horizon_{horizon}"
            results["horizons"][horizon_key] = {}
            
            # Indice de base pour cet horizon dans y_test
            # Ici, on suppose que y_test[h_idx] correspond à 'direction'
            y_true_direction = y_test[h_idx]
            y_pred_direction = (predictions[h_idx] > 0.5).astype(int).flatten()
            
            from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
            accuracy = accuracy_score(y_true_direction, y_pred_direction)
            precision = precision_score(y_true_direction, y_pred_direction, zero_division=0)
            recall = recall_score(y_true_direction, y_pred_direction, zero_division=0)
            f1 = f1_score(y_true_direction, y_pred_direction, zero_division=0)
            
            results["horizons"][horizon_key]["direction"] = {
                "accuracy": float(accuracy),
                "precision": float(precision),
                "recall": float(recall),
                "f1_score": float(f1),
                "confusion_matrix": confusion_matrix(y_true_direction, y_pred_direction).tolist()
            }
            
            # ...existing code to calculer d'autres métriques si applicable...
        
        return results