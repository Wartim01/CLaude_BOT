"""
Utilitaire pour vérifier la cohérence des caractéristiques dans le pipeline d'entraînement.
Permet de détecter les problèmes de compatibilité entre l'entraînement et la prédiction.
"""
import sys
import os
import argparse
import pandas as pd
import numpy as np
import json
from pathlib import Path
from typing import Dict, List, Tuple, Union

# Ajouter le répertoire racine au path pour pouvoir importer les modules du projet
root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, root_dir)

from ai.models.feature_engineering import FeatureEngineering
from data.data_manager import load_market_data
from utils.logger import setup_logger
from config.feature_config import get_optimal_feature_count

logger = setup_logger("feature_consistency_check")

def check_feature_consistency(training_data: pd.DataFrame, prediction_data: pd.DataFrame,
                             expected_feature_count: int = None) -> Dict:
    """
    Vérifie la cohérence des caractéristiques entre les jeux de données d'entraînement et de prédiction
    
    Args:
        training_data: DataFrame avec les données d'entraînement
        prediction_data: DataFrame avec les données de prédiction
        expected_feature_count: Nombre de caractéristiques attendu
        
    Returns:
        Dictionnaire avec les résultats de la vérification
    """
    # Utiliser la configuration centralisée si aucune valeur n'est fournie
    if expected_feature_count is None:
        expected_feature_count = get_optimal_feature_count()
        
    logger.info(f"Vérification de la cohérence des caractéristiques (nombre attendu: {expected_feature_count})")
    
    # Créer une instance de FeatureEngineering
    fe = FeatureEngineering(save_scalers=False, expected_feature_count=expected_feature_count)
    
    # Générer les caractéristiques pour les données d'entraînement
    try:
        train_features = fe.create_features(training_data, enforce_consistency=True)
        logger.info(f"Caractéristiques générées pour l'entraînement: {train_features.shape}")
        
        # Vérifier le nombre de colonnes
        if train_features.shape[1] != expected_feature_count:
            logger.warning(f"Le nombre de caractéristiques d'entraînement ({train_features.shape[1]}) "
                          f"ne correspond pas au nombre attendu ({expected_feature_count})")
    except Exception as e:
        logger.error(f"Erreur lors de la génération des caractéristiques d'entraînement: {str(e)}")
        return {"success": False, "error": str(e)}
    
    # Générer les caractéristiques pour les données de prédiction
    try:
        predict_features = fe.create_features(prediction_data, enforce_consistency=True)
        logger.info(f"Caractéristiques générées pour la prédiction: {predict_features.shape}")
    except Exception as e:
        logger.error(f"Erreur lors de la génération des caractéristiques de prédiction: {str(e)}")
        return {"success": False, "error": str(e)}
    
    # Vérifier la cohérence
    consistency_results = {
        "success": True,
        "training_columns": train_features.shape[1],
        "prediction_columns": predict_features.shape[1],
        "columns_count_match": train_features.shape[1] == predict_features.shape[1],
        "column_names_match": all(c1 == c2 for c1, c2 in zip(train_features.columns, predict_features.columns)),
        "issues": []
    }
    
    # Analyser les différences si nécessaire
    if not consistency_results["columns_count_match"]:
        consistency_results["issues"].append(
            f"Le nombre de colonnes diffère: {train_features.shape[1]} vs {predict_features.shape[1]}"
        )
    
    if not consistency_results["column_names_match"]:
        # Trouver les colonnes différentes
        cols_train = set(train_features.columns)
        cols_predict = set(predict_features.columns)
        
        missing_in_predict = cols_train - cols_predict
        extra_in_predict = cols_predict - cols_train
        
        if missing_in_predict:
            consistency_results["issues"].append(
                f"Colonnes manquantes dans les données de prédiction: {missing_in_predict}"
            )
        
        if extra_in_predict:
            consistency_results["issues"].append(
                f"Colonnes supplémentaires dans les données de prédiction: {extra_in_predict}"
            )
    
    # Vérifier les statistiques de base des colonnes communes
    common_cols = list(set(train_features.columns) & set(predict_features.columns))
    stats_diff = {}
    
    for col in common_cols:
        train_mean = train_features[col].mean()
        train_std = train_features[col].std()
        predict_mean = predict_features[col].mean()
        
        # Vérifier si la moyenne des données de prédiction est très différente de celle d'entraînement
        if train_std > 0 and abs(predict_mean - train_mean) > 3 * train_std:
            stats_diff[col] = {
                "train_mean": float(train_mean),
                "predict_mean": float(predict_mean),
                "train_std": float(train_std),
                "deviation_sigmas": float(abs(predict_mean - train_mean) / train_std)
            }
    
    if stats_diff:
        consistency_results["statistical_differences"] = stats_diff
        consistency_results["issues"].append(
            f"Différences statistiques significatives détectées pour {len(stats_diff)} colonnes"
        )
    
    # Résultat global
    consistency_results["consistent"] = (
        consistency_results["columns_count_match"] and 
        consistency_results["column_names_match"] and 
        len(stats_diff) == 0
    )
    
    return consistency_results

def save_consistency_report(results: Dict, output_path: str = None) -> str:
    """
    Sauvegarde un rapport de cohérence des caractéristiques
    
    Args:
        results: Résultats de la vérification de cohérence
        output_path: Chemin du fichier de sortie (si None, un nom est généré automatiquement)
        
    Returns:
        Chemin du fichier de rapport
    """
    if output_path is None:
        from datetime import datetime
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_dir = os.path.join(root_dir, "data", "reports")
        os.makedirs(report_dir, exist_ok=True)
        output_path = os.path.join(report_dir, f"feature_consistency_{timestamp}.json")
    
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    logger.info(f"Rapport de cohérence sauvegardé: {output_path}")
    return output_path

def verify_pipeline_consistency(symbol: str = "BTCUSDT", timeframe: str = "1h", 
                              train_size_ratio: float = 0.8,
                              expected_feature_count: int = 78) -> Dict:
    """
    Vérifie la cohérence complète du pipeline d'ingénierie des caractéristiques
    
    Args:
        symbol: Symbole de trading
        timeframe: Unité de temps
        train_size_ratio: Ratio pour la taille des données d'entraînement
        expected_feature_count: Nombre de caractéristiques attendu
        
    Returns:
        Résultats de la vérification
    """
    try:
        # Charger les données
        data_loader = load_market_data()
        data = data_loader.load_historical_data(symbol, timeframe)
        
        if data is None or len(data) < 100:
            logger.error("Données insuffisantes pour la vérification")
            return {"success": False, "error": "Données insuffisantes"}
        
        # Diviser en ensembles d'entraînement et de prédiction
        train_size = int(len(data) * train_size_ratio)
        train_data = data.iloc[:train_size]
        prediction_data = data.iloc[train_size:]
        
        logger.info(f"Données divisées: {len(train_data)} lignes pour l'entraînement, {len(prediction_data)} pour la prédiction")
        
        # Vérifier la cohérence
        consistency_results = check_feature_consistency(
            train_data, prediction_data, expected_feature_count
        )
        
        # Sauvegarder le rapport
        save_consistency_report(consistency_results)
        
        return consistency_results
        
    except Exception as e:
        logger.error(f"Erreur lors de la vérification de cohérence du pipeline: {str(e)}")
        return {"success": False, "error": str(e)}

def main():
    """Point d'entrée principal"""
    parser = argparse.ArgumentParser(description="Vérifier la cohérence des caractéristiques")
    parser.add_argument("--symbol", type=str, default="BTCUSDT", help="Symbole de trading")
    parser.add_argument("--timeframe", type=str, default="1h", help="Unité de temps")
    parser.add_argument("--train-ratio", type=float, default=0.8, help="Ratio des données d'entraînement")
    parser.add_argument("--features", type=int, default=78, help="Nombre de caractéristiques attendu")
    parser.add_argument("--report", type=str, help="Chemin pour le rapport de sortie")
    
    args = parser.parse_args()
    
    logger.info("Démarrage de la vérification de cohérence des caractéristiques")
    
    # Option 1: Vérification complète du pipeline
    consistency_results = verify_pipeline_consistency(
        symbol=args.symbol,
        timeframe=args.timeframe,
        train_size_ratio=args.train_ratio,
        expected_feature_count=args.features
    )
    
    # Sauvegarder le rapport si spécifié
    if args.report:
        save_consistency_report(consistency_results, args.report)
    
    # Afficher le résultat
    if consistency_results.get("consistent", False):
        print("\n✅ Pipeline cohérent: les caractéristiques sont identiques entre l'entraînement et la prédiction.")
    else:
        print("\n❌ Problèmes de cohérence détectés dans le pipeline de caractéristiques:")
        for issue in consistency_results.get("issues", []):
            print(f"  - {issue}")

if __name__ == "__main__":
    main()
