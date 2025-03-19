"""
Exemple d'utilisation de l'auto-optimisation du nombre de caractéristiques.
"""
import sys
import os
import pandas as pd
import numpy as np

# Ajouter le répertoire racine au path
root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, root_dir)

from ai.models.feature_engineering import FeatureEngineering
from ai.models.lstm_model import LSTMModel
from data.data_manager import load_market_data
from utils.logger import setup_logger
from utils.feature_optimizer import run_feature_optimization

logger = setup_logger("optimize_features_example")

def main():
    """Point d'entrée principal"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Exemple d'auto-optimisation des caractéristiques")
    parser.add_argument("--symbol", type=str, default="BTCUSDT", help="Symbole de trading")
    parser.add_argument("--timeframe", type=str, default="1h", help="Unité de temps")
    parser.add_argument("--min", type=int, default=30, help="Nombre minimum de caractéristiques")
    parser.add_argument("--max", type=int, default=100, help="Nombre maximum de caractéristiques")
    
    args = parser.parse_args()
    
    # Exécuter l'optimisation via l'utilitaire existant
    optimal_count = run_feature_optimization(
        data_path=None,  # Utilise data_loader
        symbol=args.symbol,
        timeframe=args.timeframe,
        min_features=args.min,
        max_features=args.max,
        step_size=10,
        plot_results=True
    )
    
    if optimal_count:
        print(f"\nNombre optimal de caractéristiques: {optimal_count}")
        print("Configuration terminée avec succès!")
    else:
        print("\nÉchec de l'optimisation. Consultez les logs pour plus de détails.")
        sys.exit(1)

if __name__ == "__main__":
    main()
