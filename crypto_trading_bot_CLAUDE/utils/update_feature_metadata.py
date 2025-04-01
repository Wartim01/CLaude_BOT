"""
Ce script régénère le fichier feature_metadata.json avec les 82 caractéristiques
définies dans FIXED_FEATURES.
"""
import os
import sys
import pandas as pd
import numpy as np
import json
from datetime import datetime

# Ajouter le répertoire parent au chemin Python
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.config import DATA_DIR
from config.feature_config import FIXED_FEATURES
from ai.models.feature_engineering import FeatureEngineering
from utils.logger import setup_logger

logger = setup_logger("update_feature_metadata")

def update_feature_metadata():
    """
    Met à jour le fichier feature_metadata.json avec les 82 caractéristiques
    """
    print("==== Mise à jour des métadonnées de caractéristiques ====")
    
    # Créer répertoire pour les scalers si nécessaire
    scalers_path = os.path.join(DATA_DIR, "models", "scalers")
    os.makedirs(scalers_path, exist_ok=True)
    
    # Charger les données pour obtenir quelques statistiques
    try:
        data_path = os.path.join(DATA_DIR, "market_data", "BTCUSDT_15m.csv")
        if not os.path.exists(data_path):
            alt_path = "c:\\Users\\timot\\OneDrive\\Bureau\\BOT TRADING BIG 2025\\crypto_trading_bot_CLAUDE\\data\\market_data\\BTCUSDT_15m.csv"
            if os.path.exists(alt_path):
                data_path = alt_path
        
        # Vérifier si le fichier existe
        if not os.path.exists(data_path):
            print("Fichier de données non trouvé. Création des métadonnées sans statistiques.")
            data = None
        else:
            data = pd.read_csv(data_path)
            print(f"Données chargées: {len(data)} lignes")
    except Exception as e:
        print(f"Erreur lors du chargement des données: {str(e)}")
        data = None
    
    # Calculer ou générer les statistiques
    feature_stats = {}
    if data is not None:
        # Générer les caractéristiques et calculer les statistiques
        fe = FeatureEngineering(save_scalers=False, expected_feature_count=len(FIXED_FEATURES))
        features = fe.create_features(data, enforce_consistency=False)
        
        # Pour chaque caractéristique dans FIXED_FEATURES
        for feature in FIXED_FEATURES:
            if feature in features.columns:
                # Calculer les statistiques réelles
                feature_stats[feature] = {
                    "mean": float(features[feature].mean()),
                    "std": float(features[feature].std()),
                    "min": float(features[feature].min()),
                    "max": float(features[feature].max()),
                    "nan_count": int(features[feature].isna().sum())
                }
            else:
                # Si la caractéristique n'est pas disponible, utiliser des valeurs par défaut
                feature_stats[feature] = {
                    "mean": 0.0,
                    "std": 1.0,
                    "min": -1.0,
                    "max": 1.0,
                    "nan_count": 0
                }
    else:
        # Si aucune donnée n'est disponible, utiliser des valeurs fictives
        for feature in FIXED_FEATURES:
            feature_stats[feature] = {
                "mean": 0.0,
                "std": 1.0,
                "min": -1.0,
                "max": 1.0,
                "nan_count": 0
            }
    
    # Créer les métadonnées
    metadata = {
        "timestamp": datetime.now().isoformat(),
        "feature_count": len(FIXED_FEATURES),
        "feature_list": FIXED_FEATURES,
        "feature_stats": feature_stats
    }
    
    # Sauvegarder les métadonnées
    metadata_path = os.path.join(scalers_path, "feature_metadata.json")
    try:
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2, default=str)
        print(f"Métadonnées sauvegardées: {metadata_path}")
        print(f"Nombre de caractéristiques: {len(FIXED_FEATURES)}")
        
        # Sauvegarder la liste des caractéristiques séparément pour référence
        features_list_path = os.path.join(scalers_path, "feature_list.json")
        with open(features_list_path, 'w') as f:
            json.dump(FIXED_FEATURES, f, indent=2)
        print(f"Liste des caractéristiques sauvegardée: {features_list_path}")
        
        return True
    except Exception as e:
        print(f"Erreur lors de la sauvegarde des métadonnées: {str(e)}")
        return False

if __name__ == "__main__":
    update_feature_metadata()
    print("\nPour vérifier si toutes les caractéristiques sont correctement utilisées, exécutez:")
    print("python utils/check_features.py")
