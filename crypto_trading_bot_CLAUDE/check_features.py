"""
Script minimal pour comparer les caractéristiques générées et celles utilisées par le modèle
"""
import os
import sys
import pandas as pd

# Ajouter le répertoire du projet au path Python
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from ai.models.feature_engineering import FeatureEngineering
from config.feature_config import FIXED_FEATURES

# Chemin vers les données
data_path = "data/market_data/BTCUSDT_15m.csv"

print(f"Vérification du fichier {data_path}...")
if not os.path.exists(data_path):
    print(f"Fichier non trouvé: {data_path}")
    sys.exit(1)

# Charger les données
print("Chargement des données...")
data = pd.read_csv(data_path)
print(f"Données chargées: {len(data)} lignes")

# Générer toutes les caractéristiques
print("Génération des caractéristiques...")
fe = FeatureEngineering(save_scalers=False)
full_features = fe.create_features(data, include_time_features=True, include_price_patterns=True, enforce_consistency=False)

# Obtenir les listes de caractéristiques
all_features = list(full_features.columns)
fixed_features = list(FIXED_FEATURES)

# Identifier les caractéristiques ignorées
ignored_features = [f for f in all_features if f not in fixed_features]

print(f"\nToutes les caractéristiques: {len(all_features)}")
print(f"Caractéristiques utilisées par le modèle: {len(fixed_features)}")
print(f"Caractéristiques ignorées: {len(ignored_features)}")

if ignored_features:
    print("\nListe des caractéristiques ignorées:")
    for i, feature in enumerate(ignored_features, 1):
        print(f"{i}. {feature}")
else:
    print("\nAucune caractéristique n'est ignorée.")
