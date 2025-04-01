"""
Script pour forcer la régénération complète des caractéristiques, en ignorant toute configuration existante.
Ce script détruit d'abord les fichiers de configuration précédents, puis régénère tout à partir de zéro.
"""
import os
import sys
import pandas as pd
import json
import shutil
from datetime import datetime

# Ajouter le répertoire parent au chemin Python
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.config import DATA_DIR
from config.feature_config import FIXED_FEATURES
from ai.models.feature_engineering import FeatureEngineering
from utils.update_feature_metadata import update_feature_metadata
from utils.regenerate_all_features import regenerate_features

def force_regenerate():
    """
    Force la régénération complète des métadonnées et features en supprimant d'abord
    les fichiers de configuration existants.
    """
    print("===== FORCE RÉGÉNÉRATION COMPLÈTE DES CARACTÉRISTIQUES =====")
    
    # Chemin vers le répertoire des scalers
    scalers_path = os.path.join(DATA_DIR, "models", "scalers")
    
    # Liste des fichiers à supprimer
    files_to_delete = [
        "feature_config.json",
        "feature_metadata.json",
        "feature_list.json",
        "lstm_standard_scaler.pkl",
        "lstm_minmax_scaler.pkl",
        "lstm_optim_standard_scaler.pkl"
    ]
    
    # Sauvegarde des fichiers existants
    backup_dir = os.path.join(scalers_path, f"backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
    if os.path.exists(scalers_path):
        os.makedirs(backup_dir, exist_ok=True)
        print(f"Sauvegarde des fichiers existants dans {backup_dir}")
        
        for file in files_to_delete:
            file_path = os.path.join(scalers_path, file)
            if os.path.exists(file_path):
                shutil.copy2(file_path, os.path.join(backup_dir, file))
                print(f"Fichier sauvegardé: {file}")
                
                # Supprimer le fichier original
                try:
                    os.remove(file_path)
                    print(f"Fichier supprimé: {file}")
                except Exception as e:
                    print(f"Erreur lors de la suppression de {file}: {str(e)}")
    
    # Vérifier le nombre actuel de caractéristiques
    print(f"\nNombre de caractéristiques dans FIXED_FEATURES: {len(FIXED_FEATURES)}")
    
    # Créer le fichier feature_metadata.json manuellement
    metadata_path = os.path.join(scalers_path, "feature_metadata.json")
    os.makedirs(os.path.dirname(metadata_path), exist_ok=True)
    
    # Mettre à jour les métadonnées puis régénérer les caractéristiques
    print("\nRégénération des métadonnées des caractéristiques...")
    update_feature_metadata()
    
    # Vérifier si le fichier a été créé correctement
    if os.path.exists(metadata_path):
        try:
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
            print(f"Fichier metadata créé avec {metadata.get('feature_count')} caractéristiques.")
        except Exception as e:
            print(f"Erreur lors de la lecture du fichier metadata: {str(e)}")
    
    # Régénérer les caractéristiques
    print("\nRégénération des caractéristiques...")
    
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
    
    # Créer l'instance de FeatureEngineering avec le bon nombre de caractéristiques
    fe = FeatureEngineering(save_scalers=True, expected_feature_count=len(FIXED_FEATURES))
    
    # Générer les caractéristiques
    print("Génération forcée des caractéristiques...")
    features = fe.create_features(data, enforce_consistency=True)
    
    print(f"Nombre de caractéristiques générées: {features.shape[1]}")
    
    # Vérifier que toutes les caractéristiques attendues sont présentes
    missing_features = [f for f in FIXED_FEATURES if f not in features.columns]
    
    if missing_features:
        print(f"\nERREUR: {len(missing_features)} caractéristiques manquantes:")
        for feature in missing_features:
            print(f"  - {feature}")
        print("\nLa régénération a échoué.")
    else:
        print(f"SUCCÈS: Toutes les {len(FIXED_FEATURES)} caractéristiques sont correctement intégrées!")
    
    print("\nExecutez maintenant la commande suivante pour vérifier l'état des caractéristiques:")
    print("python utils/check_features.py")
    
    return True

if __name__ == "__main__":
    force_regenerate()
