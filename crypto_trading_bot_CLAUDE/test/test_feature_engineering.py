import sys, os
# Insert project root (one level up from test/) into sys.path so that "ai" can be imported
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import unittest
import pandas as pd
import numpy as np
from ai.models.feature_engineering import FeatureEngineering
from config.feature_config import get_optimal_feature_count, update_optimal_feature_count
from utils.logger import setup_logger

logger = setup_logger("test_feature_engineering")

class TestFeatureEngineering(unittest.TestCase):
    def setUp(self):
        # Créer un DataFrame synthétique avec 100 lignes et colonnes OHLCV
        dates = pd.date_range(start="2021-01-01", periods=100, freq="min")
        data = {
            "open": np.random.uniform(30000, 40000, 100),
            "high": np.random.uniform(40000, 50000, 100),
            "low": np.random.uniform(30000, 40000, 100),
            "close": np.random.uniform(30000, 50000, 100),
            "volume": np.random.uniform(100, 1000, 100)
        }
        self.df = pd.DataFrame(data, index=dates)
    
    def test_create_features(self):
        fe = FeatureEngineering(save_scalers=False, expected_feature_count=78)
        # Forcer l’harmonisation en passant enforce_consistency=True
        df_features = fe.create_features(self.df, enforce_consistency=True)
        
        # Vérifier qu'il n'y a aucune valeur manquante
        self.assertFalse(df_features.isnull().values.any(), "Il y a des valeurs manquantes dans les features")
        
        # Vérifier que le nombre de colonnes est exactement celui attendu (78)
        self.assertEqual(df_features.shape[1], 78,
                         f"Nombre de features attendu: 78, obtenu: {df_features.shape[1]}")
        
        # Afficher le nombre de features utilisé (via log)
        logger.info(f"Nombre de features généré: {df_features.shape[1]}")
    
    def test_optimize_feature_count(self):
        """Vérifie que l'optimisation du nombre de caractéristiques fonctionne correctement"""
        # Créer un DataFrame plus grand pour avoir des résultats d'optimisation significatifs
        dates = pd.date_range(start="2021-01-01", periods=300, freq="15T")  # Utilisation de 'min' au lieu de 'T'
        data = {
            "open": np.random.uniform(30000, 40000, 300),
            "high": np.random.uniform(40000, 50000, 300),
            "low": np.random.uniform(30000, 40000, 300),
            "close": np.random.uniform(30000, 50000, 300),
            "volume": np.random.uniform(100, 1000, 300)
        }
        df = pd.DataFrame(data, index=dates)
        
        # Ajouter une variable cible pour le test d'optimisation
        df['price_direction'] = (df['close'].shift(-1) > df['close']).astype(int)
        
        # Créer une instance de FeatureEngineering avec auto-optimisation désactivée
        fe = FeatureEngineering(save_scalers=False, auto_optimize=True)
        fe = FeatureEngineering(save_scalers=False, auto_optimize=False)
        
        # Exécuter l'optimisation avec des paramètres réduits pour accélérer le test
        optimal_count = fe.optimize_feature_count(
            df, 
            min_features=30, 
            max_features=60, 
            step_size=10,
            cv_folds=2
        )
        
        # Vérifier que l'optimisation a retourné un nombre de caractéristiques valide
        self.assertGreaterEqual(optimal_count, 30, "Le nombre optimal de caractéristiques est trop faible")
        self.assertLessEqual(optimal_count, 60, "Le nombre optimal de caractéristiques est trop élevé")
        
        # Vérifier que le nombre optimal a été enregistré dans l'instance
        self.assertEqual(fe.optimal_feature_count, optimal_count, 
                        "Le nombre optimal de caractéristiques n'a pas été correctement stocké")
        
        # Remplacer les paramètres de l'instance pour forcer la cohérence
        fe.expected_feature_count = optimal_count
        
        # Créer un nouveau jeu de caractéristiques en forçant le nombre optimal
        optimized_features = fe.create_features(
            df, 
            enforce_consistency=True, 
            force_feature_count=optimal_count
        )
        
        # Vérifier que le nombre de caractéristiques générées correspond exactement à l'optimal
        self.assertEqual(optimized_features.shape[1], optimal_count, 
                        f"Nombre de caractéristiques attendu: {optimal_count}, obtenu: {optimized_features.shape[1]}")
        
        logger.info(f"Test d'optimisation réussi: nombre optimal de caractéristiques = {optimal_count}")
        
    def test_load_feature_configuration(self):
        # Sauvegarder la valeur actuelle pour la restaurer plus tard
        original_feature_count = get_optimal_feature_count()
        
        # Définir une valeur de test pour le test
        test_feature_count = 42
        update_optimal_feature_count(test_feature_count)
        
        # Créer une instance de FeatureEngineering explicitement avec la même valeur
        # pour éviter qu'elle ne charge une autre valeur du système de configuration
        fe = FeatureEngineering(save_scalers=True, auto_optimize=True, expected_feature_count=test_feature_count)
        
        # Générer une configuration factice et la sauvegarder
        fe.optimal_feature_count = test_feature_count
        fe.fixed_features = [f"feature_{i}" for i in range(test_feature_count)]
        fe.feature_importances = {f"feature_{i}": (100-i)/100 for i in range(test_feature_count)}
        
        # Forcer l'écriture des métadonnées avec cette configuration
        df_features = pd.DataFrame({f"feature_{i}": range(10) for i in range(test_feature_count)})
        fe._save_feature_metadata(df_features)
        
        # Sauvegarder la configuration
        result = fe.optimize_and_configure(self.df, save_config=True)
        
        # Créer une nouvelle instance avec le même paramètre explicite
        fe_new = FeatureEngineering(save_scalers=True, auto_optimize=True, expected_feature_count=test_feature_count)
        loaded = fe_new.load_feature_configuration()
        
        # Vérifier que la configuration a été chargée correctement
        self.assertTrue(loaded, "La configuration n'a pas été chargée correctement")
        self.assertEqual(fe_new.optimal_feature_count, test_feature_count, 
                        "Le nombre optimal de caractéristiques n'a pas été chargé correctement")
        self.assertIsNotNone(fe_new.fixed_features, "La liste des caractéristiques n'a pas été chargée")
        self.assertEqual(len(fe_new.fixed_features), test_feature_count, 
                        "Le nombre de caractéristiques chargées ne correspond pas")
        self.assertIsNotNone(fe_new.feature_importances, "Les importances des caractéristiques n'ont pas été chargées")
        
        logger.info(f"Test de chargement de configuration réussi: {fe_new.optimal_feature_count} caractéristiques")
        
        # Restaurer la valeur d'origine
        update_optimal_feature_count(original_feature_count)
    
    def test_create_features_with_centralized_config(self):
        """Vérifie que la création de caractéristiques utilise la configuration centralisée"""
        # Sauvegarder la valeur actuelle pour la restaurer plus tard
        original_feature_count = get_optimal_feature_count()
        
        # Définir une nouvelle valeur de test
        test_feature_count = 50
        update_optimal_feature_count(test_feature_count)
        
        # Créer une instance sans spécifier de nombre de caractéristiques
        fe = FeatureEngineering(save_scalers=False)
        
        # Vérifier que la valeur a été récupérée depuis la configuration centralisée
        self.assertEqual(fe.expected_feature_count, test_feature_count,
                        f"Valeur attendue: {test_feature_count}, obtenue: {fe.expected_feature_count}")
        
        # Générer les caractéristiques
        features = fe.create_features(self.df)
        
        # Vérifier le nombre de caractéristiques générées
        self.assertEqual(features.shape[1], test_feature_count,
                        f"Nombre de caractéristiques attendu: {test_feature_count}, obtenu: {features.shape[1]}")
        
        # Restaurer la valeur d'origine
        update_optimal_feature_count(original_feature_count)

    def test_feature_consistency_across_calls(self):
        """Vérifie que les caractéristiques sont cohérentes entre différents appels"""
        # Créer deux DataFrames avec des données légèrement différentes
        dates1 = pd.date_range(start="2021-01-01", periods=100, freq="15T")
        dates2 = pd.date_range(start="2021-02-01", periods=100, freq="15T")
        
        data1 = {
            "open": np.random.uniform(30000, 40000, 100),
            "high": np.random.uniform(40000, 50000, 100),
            "low": np.random.uniform(30000, 40000, 100),
            "close": np.random.uniform(30000, 50000, 100),
            "volume": np.random.uniform(100, 1000, 100)
        }
        
        data2 = {
            "open": np.random.uniform(35000, 45000, 100),  # Différente plage de valeurs
            "high": np.random.uniform(45000, 55000, 100),
            "low": np.random.uniform(35000, 45000, 100),
            "close": np.random.uniform(35000, 55000, 100),
            "volume": np.random.uniform(200, 2000, 100)    # Volume plus élevé
        }
        
        df1 = pd.DataFrame(data1, index=dates1)
        df2 = pd.DataFrame(data2, index=dates2)
        
        # Créer l'instance de FeatureEngineering
        fe = FeatureEngineering(save_scalers=False, expected_feature_count=78)
        
        # Premier appel pour établir la configuration
        features1 = fe.create_features(df1, enforce_consistency=True)
        
        # Vérifier que la liste des caractéristiques est stockée
        self.assertIsNotNone(fe.fixed_features, "La liste de caractéristiques devrait être stockée")
        self.assertEqual(len(fe.fixed_features), 78, "La liste de caractéristiques devrait contenir 78 éléments")
        
        # Deuxième appel avec des données différentes
        features2 = fe.create_features(df2, enforce_consistency=True)
        
        # Vérifier la cohérence entre les deux appels
        self.assertEqual(features1.shape[1], features2.shape[1], 
                         "Le nombre de colonnes doit être identique entre les appels")
        
        # Vérifier que toutes les colonnes du premier appel sont présentes dans le même ordre
        all_columns_match = all(features1.columns[i] == features2.columns[i] for i in range(len(features1.columns)))
        self.assertTrue(all_columns_match, "Les colonnes doivent être dans le même ordre entre les appels")
        
        # Vérifier la fonction de vérification de cohérence
        consistency_check = fe.verify_feature_consistency(df2)
        self.assertTrue(consistency_check["consistent"], 
                        "La vérification de cohérence doit réussir pour des données valides")
        
    def test_feature_persistence(self):
        """Vérifie que la configuration des caractéristiques persiste après sauvegarde/chargement"""
        # Créer une instance avec auto-optimisation
        fe1 = FeatureEngineering(save_scalers=True, auto_optimize=True)
        
        # Générer des caractéristiques et sauvegarder la configuration
        features = fe1.create_features(self.df, enforce_consistency=True)
        fe1._save_feature_metadata(features)
        
        # Créer une nouvelle instance
        fe2 = FeatureEngineering(save_scalers=True)
        
        # Charger la configuration
        loaded = fe2.load_feature_configuration()
        self.assertTrue(loaded, "La configuration devrait être chargée avec succès")
        
        # Vérifier que les listes de caractéristiques sont identiques
        self.assertEqual(len(fe1.fixed_features), len(fe2.fixed_features),
                         "Le nombre de caractéristiques doit être identique après chargement")
        
        self.assertListEqual(fe1.fixed_features, fe2.fixed_features,
                            "Les listes de caractéristiques doivent être identiques après chargement")
        
        # Générer des caractéristiques avec la nouvelle instance
        features2 = fe2.create_features(self.df, enforce_consistency=True)
        
        # Vérifier que les colonnes sont identiques
        self.assertListEqual(list(features.columns), list(features2.columns),
                            "Les colonnes doivent être identiques après rechargement de la configuration")
    
    def test_feature_impact_evaluation(self):
        """Vérifie que l'évaluation de l'impact des caractéristiques fonctionne correctement"""
        # Créer un DataFrame plus grand pour l'évaluation
        dates = pd.date_range(start="2021-01-01", periods=200, freq="15T")
        data = {
            "open": np.random.uniform(30000, 40000, 200),
            "high": np.random.uniform(40000, 50000, 200),
            "low": np.random.uniform(30000, 40000, 200),
            "close": np.random.uniform(30000, 50000, 200),
            "volume": np.random.uniform(100, 1000, 200)
        }
        df = pd.DataFrame(data, index=dates)
        
        # Ajouter une variable cible
        df['target'] = (df['close'].shift(-1) > df['close']).astype(int)
        
        # Créer l'instance et évaluer l'impact
        fe = FeatureEngineering(save_scalers=False)
        impact_results = fe.evaluate_feature_impact(df, target_variable='target')
        
        # Vérifications de base
        self.assertIsInstance(impact_results, pd.DataFrame, "Les résultats devraient être un DataFrame")
        self.assertGreater(len(impact_results), 0, "Le DataFrame des résultats ne devrait pas être vide")
        
        # Vérifier que les colonnes attendues sont présentes
        expected_columns = ['feature', 'correlation', 'abs_correlation', 'importance', 'combined_score']
        self.assertTrue(all(col in impact_results.columns for col in expected_columns),
                       "Toutes les colonnes attendues devraient être présentes dans les résultats")
        
        # Vérifier que les caractéristiques sont triées par importance
        self.assertTrue(
            all(impact_results['importance'].iloc[i] >= impact_results['importance'].iloc[i+1] 
                for i in range(len(impact_results)-1)),
            "Les caractéristiques devraient être triées par ordre d'importance décroissant"
        )
        
        # Vérifier que les importances ont été stockées dans l'instance
        self.assertIsNotNone(fe.feature_importances, "Les importances devraient être stockées dans l'instance")
        self.assertEqual(len(fe.feature_importances), len(impact_results),
                        "Le nombre d'importances stockées devrait correspondre au nombre de caractéristiques")

def test_feature_robustness_with_missing_data(self):
    """Vérifie que le pipeline gère correctement les données manquantes"""
    # Créer un DataFrame avec des données manquantes
    dates = pd.date_range(start="2021-01-01", periods=100, freq="15T")
    data = {
        "open": np.random.uniform(30000, 40000, 100),
        "high": np.random.uniform(40000, 50000, 100),
        "low": np.random.uniform(30000, 40000, 100),
        "close": np.random.uniform(30000, 50000, 100),
        "volume": np.random.uniform(100, 1000, 100)
    }
    df = pd.DataFrame(data, index=dates)
    # Introduire des valeurs manquantes (~5% pour chaque colonne)
    for col in df.columns:
        mask = np.random.random(len(df)) < 0.05
        df.loc[mask, col] = np.nan
    # Créer l'instance et générer les caractéristiques
    fe = FeatureEngineering(save_scalers=False, expected_feature_count=78)
    df_features = fe.create_features(df, enforce_consistency=True)
    # Vérifier qu'il n'y a pas de valeurs manquantes dans le résultat
    self.assertFalse(df_features.isnull().values.any(), 
                     "Le DataFrame résultant ne devrait pas contenir de valeurs manquantes")
    # Vérifier que le nombre de colonnes est correct
    self.assertEqual(df_features.shape[1], 78,
                     f"Nombre de features attendu: 78, obtenu: {df_features.shape[1]}")
