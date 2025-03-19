import unittest
import os
import json
import pandas as pd
import numpy as np
from datetime import datetime
from ai.models.feature_engineering import FeatureEngineering
from hyperparameter_search import LSTMHyperparameterOptimizer
from utils.logger import setup_logger

logger = setup_logger("test_hyperparameter_search")

class TestHyperparameterSearch(unittest.TestCase):
    def setUp(self):
        # Créer un DataFrame synthétique avec les colonnes OHLCV sur 120 lignes
        dates = pd.date_range(start="2021-01-01", periods=120, freq="15T")
        data = {
            "open": np.random.uniform(100, 200, 120),
            "high": np.random.uniform(150, 250, 120),
            "low": np.random.uniform(80, 150, 120),
            "close": np.random.uniform(100, 200, 120),
            "volume": np.random.uniform(1000, 5000, 120)
        }
        self.df = pd.DataFrame(data, index=dates)
        # Diviser en données d'entraînement et de validation
        train_size = int(len(self.df) * 0.7)
        self.train_data = self.df.iloc[:train_size]
        self.val_data = self.df.iloc[train_size:]
        # Nettoyer le répertoire d'optimisation (optionnel)
        self.opt_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "models", "optimization")
        if not os.path.exists(self.opt_dir):
            os.makedirs(self.opt_dir, exist_ok=True)

    def test_optimization_returns_positive_f1_and_saves_json(self):
        optimizer = LSTMHyperparameterOptimizer(
            train_data=self.train_data,
            val_data=self.val_data,
            symbol="TEST",
            timeframe="15m"
        )
        # Lancer l'optimisation avec un nombre réduit d'essais
        best_params = optimizer.run_optimization(n_trials=5, timeout=60)
        # Vérifier que le best_params retourné est un dictionnaire non vide
        self.assertIsInstance(best_params, dict, "best_params doit être un dictionnaire")
        # Vérifier que la valeur F1 (objective) obtenue est supérieure à 0
        f1 = optimizer.trials_history[-1]["f1_score"] if optimizer.trials_history else 0
        self.assertGreater(f1, 0, "Le F1 score doit être supérieur à 0")
        # Vérifier qu'un fichier JSON avec les meilleurs paramètres a été créé
        json_files = [f for f in os.listdir(optimizer.output_dir) if f.startswith("best_params_") and f.endswith(".json")]
        self.assertTrue(json_files, "Aucun fichier JSON de meilleurs hyperparamètres n'a été trouvé")
        # Charger le premier fichier JSON et vérifier son contenu
        best_json_path = os.path.join(optimizer.output_dir, json_files[0])
        with open(best_json_path, 'r') as f:
            data = json.load(f)
        self.assertIn("best_params", data, "Le fichier JSON doit contenir 'best_params'")

if __name__ == "__main__":
    unittest.main()