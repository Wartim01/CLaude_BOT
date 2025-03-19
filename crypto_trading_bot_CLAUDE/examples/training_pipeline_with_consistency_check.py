"""
Pipeline d'entraînement avec vérification de cohérence des caractéristiques.
"""
import sys
import os
import argparse

# Ajouter le répertoire racine au path
root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, root_dir)

from utils.feature_consistency_check import verify_pipeline_consistency
from utils.logger import setup_logger

logger = setup_logger("training_pipeline_example")

def main():
    """Point d'entrée principal"""
    parser = argparse.ArgumentParser(description="Vérification de cohérence des caractéristiques")
    parser.add_argument("--symbol", type=str, default="BTCUSDT", help="Symbole de trading")
    parser.add_argument("--timeframe", type=str, default="1h", help="Unité de temps")
    
    args = parser.parse_args()
    
    # Vérifier la cohérence du pipeline
    results = verify_pipeline_consistency(
        symbol=args.symbol,
        timeframe=args.timeframe
    )
    
    # Afficher le résultat
    if results.get("consistent", False):
        print("\n✅ Pipeline cohérent: caractéristiques identiques entre entraînement et prédiction.")
    else:
        print("\n❌ Problèmes de cohérence détectés:")
        for issue in results.get("issues", []):
            print(f"  - {issue}")

if __name__ == "__main__":
    main()
