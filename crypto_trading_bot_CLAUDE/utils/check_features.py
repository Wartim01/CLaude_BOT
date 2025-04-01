"""
Script minimal pour vérifier que toutes les caractéristiques sont bien intégrées dans le modèle
"""
import os
import sys
import pandas as pd

# Ajouter le répertoire du projet au path Python
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ai.models.feature_engineering import FeatureEngineering
from config.feature_config import FIXED_FEATURES

def check_features():
    """Vérifie que toutes les caractéristiques sont correctement intégrées"""
    # Chemin vers les données
    data_path = "data/market_data/BTCUSDT_15m.csv"
    
    print(f"Vérification de l'intégration des caractéristiques avec {data_path}...")
    if not os.path.exists(data_path):
        alt_path = os.path.join("c:/Users/timot/OneDrive/Bureau/BOT TRADING BIG 2025/crypto_trading_bot_CLAUDE", data_path)
        if os.path.exists(alt_path):
            data_path = alt_path
            print(f"Chemin alternatif utilisé: {data_path}")
        else:
            print(f"Erreur: Fichier non trouvé. Ni {data_path} ni {alt_path} n'existent.")
            sys.exit(1)
    
    # Charger les données
    print("Chargement des données...")
    data = pd.read_csv(data_path)
    print(f"Données chargées: {len(data)} lignes")
    
    # Créer l'instance de FeatureEngineering
    print("Création de l'instance FeatureEngineering...")
    fe = FeatureEngineering(save_scalers=False)
    
    # Générer toutes les caractéristiques
    print("Génération des caractéristiques avec enforce_consistency=False...")
    full_features = fe.create_features(data, enforce_consistency=False)
    
    # Générer les caractéristiques harmonisées
    print("Génération des caractéristiques avec enforce_consistency=True...")
    harmonized_features = fe.create_features(data, enforce_consistency=True)
    
    # Comparer les résultats
    print("\nRésultats:")
    print(f"- Nombre de caractéristiques définies dans FIXED_FEATURES: {len(FIXED_FEATURES)}")
    print(f"- Nombre de caractéristiques générées sans harmonisation: {full_features.shape[1]}")
    print(f"- Nombre de caractéristiques générées avec harmonisation: {harmonized_features.shape[1]}")
    
    # Vérifier que toutes les caractéristiques FIXED_FEATURES sont présentes
    missing_features = [f for f in FIXED_FEATURES if f not in harmonized_features.columns]
    if missing_features:
        print(f"\nERREUR: {len(missing_features)} caractéristiques définies manquantes:")
        for feature in missing_features:
            print(f"  - {feature}")
    else:
        print("\nSUCCÈS: Toutes les caractéristiques définies dans FIXED_FEATURES sont présentes.")
    
    # Vérifier que toutes les caractéristiques générées sont maintenant conservées
    ignored_features = [f for f in full_features.columns if f not in harmonized_features.columns]
    if ignored_features:
        print(f"\nAttention: {len(ignored_features)} caractéristiques encore ignorées:")
        for feature in ignored_features:
            print(f"  - {feature}")
    else:
        print("\nSUCCÈS: Toutes les caractéristiques générées sont désormais conservées.")

if __name__ == "__main__":
    check_features()
