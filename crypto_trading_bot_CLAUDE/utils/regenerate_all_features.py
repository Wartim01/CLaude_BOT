"""
Script complet pour régénérer toutes les caractéristiques et valider leur intégration
correcte dans le modèle.
"""
import os
import sys
import pandas as pd
import subprocess
import time

# Ajouter le répertoire parent au chemin Python
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.update_feature_metadata import update_feature_metadata
from config.feature_config import FIXED_FEATURES
from ai.models.feature_engineering import FeatureEngineering
from utils.logger import setup_logger

logger = setup_logger("regenerate_all_features")

def regenerate_features():
    """
    Régénère toutes les métadonnées et features pour corriger les problèmes
    """
    print("===== Régénération complète des caractéristiques =====")
    
    # Étape 1: Mettre à jour les métadonnées des caractéristiques
    print("\n1. Mise à jour des métadonnées des caractéristiques...")
    update_result = update_feature_metadata()
    
    if not update_result:
        print("Échec de la mise à jour des métadonnées. Abandon.")
        return False
    
    # Étape 2: Charger les données
    print("\n2. Chargement des données de test...")
    try:
        data_path = os.path.join("data", "market_data", "BTCUSDT_15m.csv")
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
    
    # Étape 3: Vérifier que FeatureEngineering utilise toutes les caractéristiques
    print(f"\n3. Vérification de l'intégration de {len(FIXED_FEATURES)} caractéristiques...")
    
    # Créer l'instance de FeatureEngineering avec le bon nombre de caractéristiques
    fe = FeatureEngineering(save_scalers=True, expected_feature_count=len(FIXED_FEATURES))
    
    # Générer les caractéristiques en forçant la cohérence
    print("Génération des caractéristiques...")
    features = fe.create_features(data, enforce_consistency=True)
    
    # Vérifier que toutes les caractéristiques attendues sont présentes
    missing_features = [f for f in FIXED_FEATURES if f not in features.columns]
    
    if missing_features:
        print(f"\nERREUR: {len(missing_features)} caractéristiques manquantes:")
        for feature in missing_features:
            print(f"  - {feature}")
        print("\nLa régénération a échoué. Essayez d'exécuter utils/check_features.py pour diagnostiquer.")
        return False
    else:
        print(f"SUCCESS: Toutes les {len(FIXED_FEATURES)} caractéristiques sont correctement intégrées!")
    
    # Étape 4: Lancer une vérification complète des caractéristiques
    print("\n4. Lancement d'une vérification complète des caractéristiques...")
    time.sleep(1)  # Pause pour laisser le temps de lire les messages
    
    # Exécuter le script check_features.py pour une vérification complète
    try:
        subprocess.run([sys.executable, "utils/check_features.py"], check=True)
    except subprocess.CalledProcessError:
        print("Échec de la vérification des caractéristiques.")
        return False
    
    print("\n===== Régénération terminée avec succès! =====")
    print("Le système est maintenant configuré pour utiliser les 82 caractéristiques.")
    
    return True

if __name__ == "__main__":
    regenerate_features()
