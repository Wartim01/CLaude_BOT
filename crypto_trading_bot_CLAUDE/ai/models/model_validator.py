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
        
        # Faire des prédictions
        predictions = self.model.model.predict(X_test)
        
        # Calculer des métriques détaillées pour chaque horizon
        results = {
            "loss": evaluation[0],
            "horizons": {}
        }
        
        # Pour chaque horizon
        for h_idx, horizon in enumerate(self.model.prediction_horizons):
            horizon_key = f"horizon_{horizon}"
            results["horizons"][horizon_key] = {}
            
            # Indice de base pour cet horizon
            base_idx = h_idx * 4
            
            # 1. Direction (classification binaire)
            y_true_direction = y_test[base_idx]
            y_pred_direction = (predictions[base_idx] > 0.5).astype(int).flatten()
            
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
            
            # 2. Volatilité (régression)
            y_true_volatility = y_test[base_idx + 1]
            y_pred_volatility = predictions[base_idx + 1].flatten()
            
            mae = np.mean(np.abs(y_true_volatility - y_pred_volatility))
            mse = np.mean((y_true_volatility - y_pred_volatility) ** 2)
            
            results["horizons"][horizon_key]["volatility"] = {
                "mae": float(mae),
                "mse": float(mse),
                "rmse": float(np.sqrt(mse))
            }
            
            # 3. Volume (