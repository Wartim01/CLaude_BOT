import unittest
import pandas as pd
import numpy as np
from ai.models.feature_engineering import FeatureEngineering

class TestDataPreparation(unittest.TestCase):
    def setUp(self):
        # Créer un DataFrame synthétique avec les colonnes OHLCV (minimum requis)
        dates = pd.date_range(start="2021-01-01", periods=100, freq="15T")
        data = {
            "open": np.random.uniform(100, 200, 100),
            "high": np.random.uniform(150, 250, 100),
            "low": np.random.uniform(80, 150, 100),
            "close": np.random.uniform(100, 200, 100),
            "volume": np.random.uniform(1000, 5000, 100)
        }
        self.df = pd.DataFrame(data, index=dates)
        # Appliquer la création puis la normalisation des features
        fe = FeatureEngineering(save_scalers=False)
        self.df_features = fe.create_features(self.df, include_time_features=True, include_price_patterns=False, enforce_consistency=False)
        self.df_scaled = fe.scale_features(self.df_features, is_training=True, method='standard', feature_group='test')
        self.fe = fe
    
    def test_multi_horizon_data_shape(self):
        sequence_length = 30
        horizons = [4]  # test avec un seul horizon
        # Appeler la fonction avec is_training=True pour générer les labels aussi
        X, y = self.fe.create_multi_horizon_data(self.df_scaled, sequence_length=sequence_length, horizons=horizons, is_training=True)
        
        # Calculer le nombre d'échantillons attendus
        n_samples_expected = len(self.df_scaled) - sequence_length - max(horizons)
        feature_dim = self.df_scaled.shape[1]
        
        # Vérifier que X est un tableau 3D
        self.assertEqual(len(X.shape), 3, "X n'est pas un tableau 3D")
        self.assertEqual(X.shape, (n_samples_expected, sequence_length, feature_dim),
                         f"Forme de X attendue: ({n_samples_expected}, {sequence_length}, {feature_dim}), obtenue: {X.shape}")
        
        # Vérifier que y est une liste/tuple contenant 4 tableaux (direction, volatilité, volume, momentum)
        self.assertTrue(isinstance(y, (list, tuple)), "y doit être une liste ou un tuple")
        self.assertEqual(len(y), 4, "Le nombre d'éléments dans y doit être 4 pour un horizon unique")
        for label in y:
            self.assertEqual(label.shape, (n_samples_expected,), f"Chaque label doit avoir la forme ({n_samples_expected},), obtenu: {label.shape}")

if __name__ == "__main__":
    unittest.main()
