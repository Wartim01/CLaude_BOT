"""
Script pour capturer et sauvegarder toutes les caractéristiques générées avant harmonisation
"""
import os
import sys
import pandas as pd
import json

# Ajouter le répertoire parent au chemin Python
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.config import DATA_DIR
from ai.models.feature_engineering import FeatureEngineering
from utils.logger import setup_logger
import config.feature_config as fc

logger = setup_logger("capture_all_features")

def capture_features():
    """
    Génère toutes les caractéristiques possibles et les sauvegarde dans FIXED_FEATURES
    """
    print("===== CAPTURE DE TOUTES LES CARACTÉRISTIQUES GÉNÉRÉES =====")
    
    # Charger les données
    try:
        data_path = os.path.join(DATA_DIR, "market_data", "BTCUSDT_15m.csv")
        if not os.path.exists(data_path):
            alt_path = "c:\\Users\\timot\\OneDrive\\Bureau\\BOT TRADING BIG 2025\\crypto_trading_bot_CLAUDE\\data\\market_data\\BTCUSDT_15m.csv"
            if os.path.exists(alt_path):
                data_path = alt_path
            else:
                print("Fichier de données non trouvé. Impossible de continuer.")
                return False
        
        data = pd.read_csv(data_path)
        print(f"Données chargées: {len(data)} lignes")
    except Exception as e:
        print(f"Erreur lors du chargement des données: {str(e)}")
        return False
    
    # Créer l'instance de FeatureEngineering
    fe = FeatureEngineering(save_scalers=False)
    
    # Générer TOUTES les caractéristiques sans harmonisation
    print("Génération de toutes les caractéristiques possibles...")
    all_features = fe.create_features(data, enforce_consistency=False)
    
    # Récupérer toutes les colonnes générées
    all_columns = list(all_features.columns)
    print(f"Nombre total de caractéristiques générées: {len(all_columns)}")
    
    # Mettre à jour FIXED_FEATURES avec toutes les colonnes générées
    print("Mise à jour de FIXED_FEATURES avec toutes les caractéristiques...")
    fc.FIXED_FEATURES = all_columns
    fc.FEATURE_COLUMNS = all_columns
    
    # Sauvegarder la liste complète des caractéristiques
    scalers_path = os.path.join(DATA_DIR, "models", "scalers")
    os.makedirs(scalers_path, exist_ok=True)
    
    all_features_path = os.path.join(scalers_path, "all_features.json")
    with open(all_features_path, 'w') as f:
        json.dump({
            "timestamp": pd.Timestamp.now().isoformat(),
            "feature_count": len(all_columns),
            "feature_list": all_columns
        }, f, indent=2)
    
    print(f"Liste de toutes les caractéristiques sauvegardée dans: {all_features_path}")
    
    # Maintenant, régénérer les caractéristiques avec enforcement pour confirmer
    print("\nRégénération des caractéristiques avec enforce_consistency=True...")
    features_enforced = fe.create_features(data, enforce_consistency=True)
    
    print(f"Nombre de caractéristiques après harmonisation: {features_enforced.shape[1]}")
    
    if len(all_columns) == features_enforced.shape[1]:
        print("\nSUCCÈS: Toutes les caractéristiques sont conservées après harmonisation!")
    else:
        print(f"\nATTENTION: {len(all_columns) - features_enforced.shape[1]} caractéristiques ont été perdues après harmonisation.")
        
        # Identifier les colonnes manquantes
        enforced_columns = list(features_enforced.columns)
        missing_columns = [col for col in all_columns if col not in enforced_columns]
        
        if missing_columns:
            print("Colonnes manquantes après harmonisation:")
            for col in missing_columns:
                print(f"  - {col}")
    
    print("\nPour utiliser toutes les caractéristiques dans votre modèle, exécutez:")
    print("python train_model.py --use_all_features")
    
    return True

if __name__ == "__main__":
    capture_features()
