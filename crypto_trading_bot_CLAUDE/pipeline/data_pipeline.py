import subprocess
import logging
import json
import re
import os
from utils.subprocess_utils import run_process, extract_metric_from_output, extract_json_metrics
from utils.path_utils import get_market_data_path, get_model_path
from utils.hyperparameter_utils import transform_optimized_params, load_optimized_parameters

logger = logging.getLogger(__name__)

def run_download_data(symbol, timeframe, start_date, end_date):
    """Exécute le script de téléchargement des données avec gestion d'erreur améliorée"""
    logger.info(f"Téléchargement des données pour {symbol} {timeframe} du {start_date} au {end_date}")
    
    # Utiliser le module path_utils pour générer le chemin de sortie
    output_path = get_market_data_path(symbol, timeframe)
    
    command = [
        "python", 
        "scripts/download_data.py",
        "--symbol", symbol,
        "--timeframe", timeframe,
        "--start_date", start_date,
        "--end_date", end_date,
        "--output_path", output_path
    ]
    
    try:
        result = run_process(
            command=command,
            process_name="téléchargement des données",
            log_output=True
        )
        logger.info(f"Téléchargement des données terminé avec succès pour {symbol} {timeframe}")
        return True, result.stdout, None
    except subprocess.CalledProcessError as e:
        error_msg = f"Échec du téléchargement des données pour {symbol} {timeframe}: {str(e)}"
        logger.error(error_msg)
        return False, e.stdout if hasattr(e, 'stdout') else None, error_msg
    except Exception as e:
        error_msg = f"Erreur lors du téléchargement des données: {str(e)}"
        logger.error(error_msg)
        return False, None, error_msg

def run_hyperparameter_search(symbol, timeframe, n_trials=30):
    """Exécute la recherche d'hyperparamètres avec gestion d'erreur améliorée"""
    logger.info(f"Lancement de la recherche d'hyperparamètres pour {symbol} {timeframe} avec {n_trials} essais")
    
    # Utiliser le module path_utils pour générer les chemins
    data_path = get_market_data_path(symbol, timeframe)
    
    command = [
        "python", 
        "scripts/hyperparameter_search.py",
        "--symbol", symbol,
        "--timeframe", timeframe,
        "--data_path", data_path,
        "--n_trials", str(n_trials)
    ]
    
    try:
        result = run_process(
            command=command,
            process_name="recherche d'hyperparamètres",
            log_output=True
        )
        
        # Extraction du meilleur F1 score
        best_f1 = extract_metric_from_output(result.stdout, r'Best F1 Score: ([0-9.]+)', default_value=0.0)
        if best_f1:
            try:
                best_f1 = float(best_f1)
                logger.info(f"Meilleur F1 score trouvé: {best_f1}")
            except ValueError:
                best_f1 = 0.0
                logger.warning("Impossible de convertir le F1 score en nombre")
        
        return True, result.stdout, {"best_f1": best_f1}
    except subprocess.CalledProcessError as e:
        error_msg = f"Échec de la recherche d'hyperparamètres pour {symbol} {timeframe}: {str(e)}"
        logger.error(error_msg)
        return False, e.stdout if hasattr(e, 'stdout') else None, error_msg
    except Exception as e:
        error_msg = f"Erreur lors de la recherche d'hyperparamètres: {str(e)}"
        logger.error(error_msg)
        return False, None, error_msg

def run_training(symbol, timeframe, version=None):
    """Exécute l'entraînement du modèle avec gestion d'erreur améliorée"""
    logger.info(f"Lancement de l'entraînement pour {symbol} {timeframe}")
    
    # Utiliser le module path_utils pour générer les chemins
    data_path = get_market_data_path(symbol, timeframe)
    model_path = get_model_path(symbol, timeframe, version)
    
    command = [
        "python", 
        "ai/models/train_model.py",
        "--symbol", symbol,
        "--timeframe", timeframe,
        "--data_path", data_path,
        "--model_path", model_path
    ]
    
    try:
        result = run_process(
            command=command,
            process_name="entraînement du modèle",
            log_output=True,
            expected_return_codes=[0, 1, 2]  # Accepter les codes 0 (succès), 1 (acceptable), 2 (échec)
        )
        
        # Extraire les métriques de la sortie
        metrics = extract_json_metrics(result.stdout)
        if not metrics:
            # Méthode de secours - utiliser le pattern F1
            f1_score = extract_metric_from_output(result.stdout, r'FINAL_F1_SCORE: ([0-9.]+)', default_value=0.0)
            if f1_score:
                try:
                    metrics = {"F1": float(f1_score)}
                except ValueError:
                    metrics = {"F1": 0.0}
        
        # Déterminer le statut en fonction du code de retour
        status_codes = {
            0: "SUCCESS",     # F1 >= 0.75
            1: "ACCEPTABLE",  # 0.65 <= F1 < 0.75
            2: "FAILED"       # F1 < 0.65
        }
        status = status_codes.get(result.returncode, "UNKNOWN")
        
        logger.info(f"Entraînement terminé avec statut: {status}")
        if metrics:
            logger.info(f"Métriques: {metrics}")
        
        return status != "FAILED", result.stdout, {"status": status, "metrics": metrics, "model_path": model_path}
    except subprocess.CalledProcessError as e:
        error_msg = f"Échec de l'entraînement pour {symbol} {timeframe}: {str(e)}"
        logger.error(error_msg)
        return False, e.stdout if hasattr(e, 'stdout') else None, {"status": "ERROR", "error": error_msg}
    except Exception as e:
        error_msg = f"Erreur lors de l'entraînement: {str(e)}"
        logger.error(error_msg)
        return False, None, {"status": "ERROR", "error": error_msg}

def run_training_with_optimized_params(symbol, timeframe, version=None):
    """Exécute l'entraînement du modèle avec des paramètres optimisés"""
    logger.info(f"Lancement de l'entraînement pour {symbol} {timeframe} avec paramètres optimisés")
    
    # Charger les paramètres optimisés
    optimized_params = load_optimized_parameters(symbol, timeframe)
    
    if not optimized_params:
        logger.warning(f"Aucun paramètre optimisé trouvé pour {symbol} {timeframe}. Utilisation des paramètres par défaut.")
        return run_training(symbol, timeframe, version)
    
    # Utiliser le module path_utils pour générer les chemins
    data_path = get_market_data_path(symbol, timeframe)
    model_path = get_model_path(symbol, timeframe, version)
    
    # Transformer les paramètres optimisés en arguments de ligne de commande
    param_args = transform_optimized_params(optimized_params)
    
    # Ajouter les arguments obligatoires
    base_args = [
        "python", 
        "ai/models/train_model.py",
        "--symbol", symbol,
        "--timeframe", timeframe,
        "--data_path", data_path,
        "--model_path", model_path
    ]
    
    # Combiner tous les arguments
    command = base_args + param_args
    
    try:
        result = run_process(
            command=command,
            process_name="entraînement avec paramètres optimisés",
            log_output=True,
            expected_return_codes=[0, 1, 2]  # Accepter les codes 0 (succès), 1 (acceptable), 2 (échec)
        )
        
        # Extraire les métriques de la sortie
        metrics = extract_json_metrics(result.stdout)
        if not metrics:
            # Méthode de secours - utiliser le pattern F1
            f1_score = extract_metric_from_output(result.stdout, r'FINAL_F1_SCORE: ([0-9.]+)', default_value=0.0)
            if f1_score:
                try:
                    metrics = {"F1": float(f1_score)}
                except ValueError:
                    metrics = {"F1": 0.0}
        
        # Déterminer le statut en fonction du code de retour
        status_codes = {
            0: "SUCCESS",     # F1 >= 0.75
            1: "ACCEPTABLE",  # 0.65 <= F1 < 0.75
            2: "FAILED"       # F1 < 0.65
        }
        status = status_codes.get(result.returncode, "UNKNOWN")
        
        logger.info(f"Entraînement terminé avec statut: {status}")
        if metrics:
            logger.info(f"Métriques: {metrics}")
        
        return status != "FAILED", result.stdout, {"status": status, "metrics": metrics, "model_path": model_path}
        
    except subprocess.CalledProcessError as e:
        error_msg = f"Échec de l'entraînement pour {symbol} {timeframe}: {str(e)}"
        logger.error(error_msg)
        return False, e.stdout if hasattr(e, 'stdout') else None, {"status": "ERROR", "error": error_msg}
    except Exception as e:
        error_msg = f"Erreur lors de l'entraînement: {str(e)}"
        logger.error(error_msg)
        return False, None, {"status": "ERROR", "error": error_msg}

def run_evaluation(symbol, timeframe, version=None):
    """Exécute l'évaluation du modèle avec gestion d'erreur améliorée"""
    logger.info(f"Lancement de l'évaluation pour {symbol} {timeframe}")
    
    # Utiliser le module path_utils pour générer les chemins
    data_path = get_market_data_path(symbol, timeframe)
    model_path = get_model_path(symbol, timeframe, version)
    
    # Vérifier que le modèle existe
    if not os.path.exists(model_path):
        error_msg = f"Modèle introuvable: {model_path}"
        logger.error(error_msg)
        return False, None, {"error": error_msg}
    
    command = [
        "python", 
        "scripts/evaluate_model.py",
        "--symbol", symbol,
        "--timeframe", timeframe,
        "--data_path", data_path,
        "--model_path", model_path,
        "--detailed"  # Option pour des métriques détaillées
    ]
    
    try:
        result = run_process(
            command=command,
            process_name="évaluation du modèle",
            log_output=True
        )
        
        # Extraire les métriques de l'évaluation
        metrics = {}
        
        # Chercher le F1 score
        f1_match = extract_metric_from_output(result.stdout, r'F1 Score: ([0-9.]+)', default_value=None)
        if f1_match:
            try:
                metrics['f1_score'] = float(f1_match)
            except ValueError:
                pass
        
        # Chercher l'accuracy
        acc_match = extract_metric_from_output(result.stdout, r'Accuracy: ([0-9.]+)', default_value=None)
        if acc_match:
            try:
                metrics['accuracy'] = float(acc_match)
            except ValueError:
                pass
        
        # Chercher la précision
        precision_match = extract_metric_from_output(result.stdout, r'Precision: ([0-9.]+)', default_value=None)
        if precision_match:
            try:
                metrics['precision'] = float(precision_match)
            except ValueError:
                pass
        
        # Chercher le recall
        recall_match = extract_metric_from_output(result.stdout, r'Recall: ([0-9.]+)', default_value=None)
        if recall_match:
            try:
                metrics['recall'] = float(recall_match)
            except ValueError:
                pass
        
        logger.info(f"Évaluation terminée avec succès pour {symbol} {timeframe}")
        if metrics:
            logger.info(f"Métriques: {metrics}")
        
        return True, result.stdout, metrics
    except subprocess.CalledProcessError as e:
        error_msg = f"Échec de l'évaluation pour {symbol} {timeframe}: {str(e)}"
        logger.error(error_msg)
        return False, e.stdout if hasattr(e, 'stdout') else None, {"error": error_msg}
    except Exception as e:
        error_msg = f"Erreur lors de l'évaluation: {str(e)}"
        logger.error(error_msg)
        return False, None, {"error": error_msg}