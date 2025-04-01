"""
Utilitaire pour analyser les caractéristiques générées et harmonisées
dans le processus de feature engineering
"""

import pandas as pd
import os
import sys
from typing import List, Dict, Tuple
import traceback

# Assurer que le répertoire parent est dans le chemin Python
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ai.models.feature_engineering import FeatureEngineering
from config.feature_config import FIXED_FEATURES
from utils.logger import setup_logger

logger = setup_logger("feature_analyzer")

def analyze_features(data_path: str, symbol: str = "BTCUSDT", timeframe: str = "15m") -> None:
    """
    Analyse les caractéristiques générées et affiche celles qui sont ignorées
    lors de l'harmonisation
    
    Args:
        data_path: Chemin vers les données de marché
        symbol: Symbole de la paire de trading
        timeframe: Intervalle de temps
    """
    try:
        # Afficher le chemin complet pour faciliter le débogage
        file_path = os.path.join(data_path, f"{symbol}_{timeframe}.csv")
        logger.info(f"Tentative de chargement du fichier: {file_path}")
        
        # Vérifier si le fichier existe
        if not os.path.exists(file_path):
            logger.error(f"Fichier non trouvé: {file_path}")
            print(f"ERREUR: Fichier non trouvé: {file_path}")
            return
        
        # Charger les données
        data = pd.read_csv(file_path)
        logger.info(f"Données chargées: {len(data)} lignes")
        print(f"Données chargées: {len(data)} lignes")
        
        # Créer une instance de FeatureEngineering
        feature_eng = FeatureEngineering(save_scalers=False)
        
        # Générer toutes les caractéristiques sans harmonisation
        logger.info("Génération de toutes les caractéristiques sans harmonisation...")
        print("Génération de toutes les caractéristiques...")
        full_features = feature_eng.create_features(
            data,
            include_time_features=True,
            include_price_patterns=True,
            enforce_consistency=False  # Désactiver l'harmonisation pour obtenir toutes les features
        )
        
        # Récupérer la liste complète des caractéristiques générées
        all_features = list(full_features.columns)
        logger.info(f"Toutes les caractéristiques générées: {len(all_features)}")
        print(f"Toutes les caractéristiques générées: {len(all_features)}")
        
        # Récupérer la liste des caractéristiques fixes (harmonisées)
        fixed_features = list(FIXED_FEATURES)
        logger.info(f"Caractéristiques harmonisées (conservées): {len(fixed_features)}")
        print(f"Caractéristiques harmonisées (conservées): {len(fixed_features)}")
        
        # Identifier les caractéristiques ignorées
        ignored_features = [f for f in all_features if f not in fixed_features]
        logger.info(f"Caractéristiques ignorées: {len(ignored_features)}")
        print(f"\nCaractéristiques ignorées: {len(ignored_features)}")
        
        if len(ignored_features) == 0:
            print("Aucune caractéristique n'est ignorée. Toutes sont conservées.")
            return
            
        # Forcer la sortie à être vidée immédiatement pour éviter les problèmes de buffering
        sys.stdout.flush()
        
        # Afficher les caractéristiques ignorées
        print("\nListe des caractéristiques ignorées:")
        for i, feature in enumerate(ignored_features, 1):
            print(f"{i}. {feature}")
            # Forcer le vidage du buffer après chaque ligne
            sys.stdout.flush()
        
        # Grouper les caractéristiques ignorées par catégorie
        categories = {}
        for feature in ignored_features:
            # Extraire la catégorie du nom de la caractéristique
            parts = feature.split('_')
            if len(parts) > 1:
                category = parts[0]
                if category not in categories:
                    categories[category] = []
                categories[category].append(feature)
            else:
                if "divers" not in categories:
                    categories["divers"] = []
                categories["divers"].append(feature)
        
        # Afficher les caractéristiques ignorées par catégorie
        print("\nCaractéristiques ignorées par catégorie:")
        for category, features in categories.items():
            print(f"\n{category.upper()} ({len(features)} caractéristiques):")
            for feature in features:
                print(f"  - {feature}")
                sys.stdout.flush()
                
        # Suggestion d'ajout de caractéristiques
        print("\n-------------------------------------------------")
        print("Pour ajouter ces caractéristiques au modèle, vous pouvez modifier")
        print("le fichier config/feature_config.py et ajouter ces caractéristiques")
        print("à la liste FIXED_FEATURES.")
        print("-------------------------------------------------")
                
    except Exception as e:
        logger.error(f"Erreur lors de l'analyse des caractéristiques: {str(e)}")
        print(f"ERREUR: {str(e)}")
        print("Détails de l'erreur:")
        traceback.print_exc()

if __name__ == "__main__":
    print("Analyseur de caractéristiques - Début de l'exécution")
    
    # Utiliser le même chemin que dans l'hyperparameter_search.py
    data_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 
                            "data", "market_data")
    
    print(f"Chemin des données: {data_path}")
    
    # Vérifier que le répertoire existe
    if not os.path.exists(data_path):
        print(f"ERREUR: Le répertoire {data_path} n'existe pas")
        # Essayer avec le chemin absolu spécifié précédemment
        data_path = "c:\\Users\\timot\\OneDrive\\Bureau\\BOT TRADING BIG 2025\\crypto_trading_bot_CLAUDE\\data\\market_data"
        print(f"Essai avec le chemin alternatif: {data_path}")
        
        if not os.path.exists(data_path):
            print(f"ERREUR: Le répertoire alternatif {data_path} n'existe pas non plus")
            sys.exit(1)
    
    try:
        analyze_features(data_path)
        print("Analyse terminée avec succès")
    except Exception as e:
        print(f"Erreur fatale: {str(e)}")
        traceback.print_exc()
