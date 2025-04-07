"""
Utilitaire pour l'optimisation des caractéristiques utilisées par le modèle LSTM.
Permet de déterminer automatiquement le nombre optimal de caractéristiques.
"""
import sys
import os
import argparse
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# Ajouter le répertoire racine au path pour pouvoir importer les modules du projet
root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, root_dir)

from ai.models.feature_engineering import FeatureEngineering
from core.data_fetcher import MarketDataFetcher
from utils.logger import setup_logger
from config.feature_config import update_optimal_feature_count, get_optimal_feature_count, DEFAULT_MIN_FEATURES, DEFAULT_MAX_FEATURES, DEFAULT_STEP_SIZE

logger = setup_logger("feature_optimizer")

def plot_optimization_results(results: dict, save_path: str = None):
    """
    Crée un graphique montrant les performances en fonction du nombre de caractéristiques
    
    Args:
        results: Dictionnaire avec les résultats d'optimisation
        save_path: Chemin pour sauvegarder le graphique (si None, affiche le graphique)
    """
    plt.figure(figsize=(12, 8))
    
    # Préparer les données pour le graphique
    feature_counts = []
    f1_scores = []
    accuracy_scores = []
    
    for count, metrics in sorted(results.items()):
        feature_counts.append(count)
        f1_scores.append(metrics['f1'])
        accuracy_scores.append(metrics['accuracy'])
    
    # Tracer les scores F1
    plt.plot(feature_counts, f1_scores, 'b-', marker='o', label='F1 Score')
    
    # Tracer les scores de précision
    plt.plot(feature_counts, accuracy_scores, 'r-', marker='x', label='Accuracy')
    
    # Déterminer le nombre optimal de caractéristiques (meilleur F1 Score)
    best_idx = f1_scores.index(max(f1_scores))
    best_count = feature_counts[best_idx]
    best_f1 = f1_scores[best_idx]
    
    plt.axvline(x=best_count, color='g', linestyle='--', label=f'Optimal: {best_count} features')
    
    plt.title('Performance vs Nombre de Caractéristiques')
    plt.xlabel('Nombre de caractéristiques')
    plt.ylabel('Score de Performance')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    
    # Annoter le point optimal
    plt.annotate(f'Optimal: {best_count} features\nF1: {best_f1:.4f}', 
                 xy=(best_count, best_f1),
                 xytext=(best_count + 5, best_f1 - 0.05),
                 arrowprops=dict(facecolor='black', shrink=0.05, width=1.5))
    
    if save_path:
        plt.tight_layout()
        plt.savefig(save_path)
        logger.info(f"Graphique sauvegardé: {save_path}")
    else:
        plt.show()

def run_feature_optimization(data_path: str, symbol: str = 'BTCUSDT', 
                          timeframe: str = '1h', min_features: int = None, 
                          max_features: int = None, step_size: int = None,
                          plot_results: bool = True):
    """
    Exécute l'optimisation complète des caractéristiques
    
    Args:
        data_path: Chemin vers les données historiques
        symbol: Symbole de trading
        timeframe: Unité de temps (ex: '1h', '15m')
        min_features: Nombre minimum de caractéristiques à tester
        max_features: Nombre maximum de caractéristiques à tester
        step_size: Pas pour tester différents nombres de caractéristiques
        plot_results: Générer un graphique des résultats
    
    Returns:
        Dictionnaire avec les résultats d'optimisation
    """
    # Utiliser la configuration centralisée si les paramètres ne sont pas spécifiés
    # The default constants are now imported at the module level.
    
    min_features = min_features if min_features is not None else DEFAULT_MIN_FEATURES
    max_features = max_features if max_features is not None else DEFAULT_MAX_FEATURES
    step_size = step_size if step_size is not None else DEFAULT_STEP_SIZE
    
    logger.info(f"Démarrage de l'optimisation des caractéristiques pour {symbol} {timeframe}")
    
    try:
        if os.path.exists(data_path):
            data = pd.read_csv(data_path, parse_dates=['timestamp'])
            data.set_index('timestamp', inplace=True)
            logger.info(f"Données chargées: {len(data)} lignes")
        else:
            logger.info(f"Fichier non trouvé: {data_path}. Tentative de chargement via DataLoader.")
            # Création du DataFetcher en mode simulé (sans API)
            data_loader = MarketDataFetcher(api_connector=None)
            logger.info(f"Mode simulé activé - génération de données factices pour {symbol} {timeframe}")
            
            # Récupération des données simulées
            market_data = data_loader.get_market_data(symbol, timeframe=timeframe, indicators=False)
            
            # Vérifier si les données du timeframe sont disponibles
            if "primary_timeframe" in market_data and "ohlcv" in market_data["primary_timeframe"]:
                data = market_data["primary_timeframe"]["ohlcv"]
                logger.info(f"Données chargées via DataLoader: {len(data)} lignes")
            else:
                # Fallback option: créer une structure pour stocker temporairement les données
                from utils.path_utils import get_market_data_path
                temp_path = get_market_data_path(symbol, timeframe)
                os.makedirs(os.path.dirname(temp_path), exist_ok=True)
                
                logger.warning(f"Données de timeframe {timeframe} non trouvées. Utilisation d'une structure vide.")
                # Créer un DataFrame vide ou charger depuis une autre source
                raise Exception(f"Impossible de charger les données pour {symbol} {timeframe}")
    except Exception as e:
        logger.error(f"Erreur lors du chargement des données: {str(e)}")
        return None
    
    # Créer l'instance de FeatureEngineering avec auto-optimisation activée
    fe = FeatureEngineering(save_scalers=True, auto_optimize=True)
    

    # Exécuter l'optimisation
    try:
        optimization_results = fe.optimize_feature_count(
            data,
            min_features=min_features,
            max_features=max_features,
            step_size=step_size,
            cv_folds=3
        )
                # Mise à jour de la configuration centralisée
        update_optimal_feature_count(optimization_results)
        

        # Configuration complète du pipeline avec le nombre optimal de caractéristiques
        full_results = fe.optimize_and_configure(data)
        

        # Générer un graphique si demandé
        if plot_results:
            output_dir = os.path.join(root_dir, "data", "reports")
            os.makedirs(output_dir, exist_ok=True)
            
            plot_path = os.path.join(output_dir, f"feature_optimization_{symbol}_{timeframe}.png")
            plot_optimization_results(fe.feature_importances, plot_path)
            
            # Générer un rapport textuel
            report_path = os.path.join(output_dir, f"feature_optimization_{symbol}_{timeframe}.txt")
            with open(report_path, 'w') as f:
                f.write(f"RAPPORT D'OPTIMISATION DES CARACTÉRISTIQUES\n")
                f.write(f"======================================\n\n")
                f.write(f"Symbole: {symbol}\n")
                f.write(f"Timeframe: {timeframe}\n")
                f.write(f"Données: {len(data)} lignes\n\n")
                f.write(f"Nombre optimal de caractéristiques: {optimization_results}\n\n")
                f.write(f"TOP 20 CARACTÉRISTIQUES PAR IMPORTANCE:\n")
                
 
                # Liste des caractéristiques triées par importance - Tri robuste
                sorted_features = sorted(fe.feature_importances.items(), 
                                        key=lambda x: x[1] if isinstance(x[1], (int, float)) else -1, 
                                        reverse=True)
                
                for i, (feature, importance) in enumerate(sorted_features[:20], 1):
                    if isinstance(importance, (int, float)):
                        f.write(f"{i:2d}. {feature}: {importance:.6f}\n")
                    else:
                        f.write(f"{i:2d}. {feature}: N/A (Invalid score: {type(importance)})\n")
                
                logger.info(f"Rapport généré: {report_path}")
        
        logger.info(f"Optimisation terminée. Nombre optimal de caractéristiques: {optimization_results}")
        return optimization_results
        
    except Exception as e:
        logger.error(f"Erreur lors de l'optimisation: {str(e)}")
        return None

if __name__ == "__main__":
    # Configuration des arguments en ligne de commande
    parser = argparse.ArgumentParser(description="Optimiser le nombre de caractéristiques pour le bot de trading")
    
    parser.add_argument("--data", type=str, help="Chemin vers les données historiques CSV")
    parser.add_argument("--symbol", type=str, default="BTCUSDT", help="Symbole de trading (défaut: BTCUSDT)")
    parser.add_argument("--timeframe", type=str, default="1h", help="Unité de temps (défaut: 1h)")
    parser.add_argument("--min", type=int, default=20, help="Nombre minimum de caractéristiques (défaut: 20)")
    parser.add_argument("--max", type=int, default=100, help="Nombre maximum de caractéristiques (défaut: 100)")
    parser.add_argument("--step", type=int, default=5, help="Pas d'incrémentation (défaut: 5)")
    parser.add_argument("--no-plot", action="store_true", help="Désactive la génération de graphique")
    
    args = parser.parse_args()
    

    # Si aucun chemin de données n'est fourni, utiliser le chemin par défaut
    if args.data is None:
        data_dir = os.path.join(root_dir, "data", "historical")
        args.data = os.path.join(data_dir, f"{args.symbol}_{args.timeframe}.csv")
    
    optimal_count = run_feature_optimization(
        data_path=args.data,
        symbol=args.symbol,
        timeframe=args.timeframe,
        min_features=args.min,
        max_features=args.max,
        step_size=args.step,
        plot_results=not args.no_plot
    )
    
    if optimal_count:
        print(f"\nNombre optimal de caractéristiques: {optimal_count}")
        print(f"Configuration terminée avec succès!")
    else:
        print("\nÉchec de l'optimisation. Consultez les logs pour plus de détails.")
        sys.exit(1)
