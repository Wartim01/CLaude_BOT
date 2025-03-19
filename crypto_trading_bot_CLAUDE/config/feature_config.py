"""
Configuration centralisée pour le module d'ingénierie des caractéristiques.
Ce fichier définit les paramètres par défaut et fournit des fonctions pour gérer
la persistance de la configuration.
"""
import os
import json
from datetime import datetime
from config.config import DATA_DIR
from utils.logger import setup_logger

logger = setup_logger("feature_config")

# Chemin vers le fichier de configuration des caractéristiques
CONFIG_FILE = os.path.join(DATA_DIR, "models", "feature_config.json")

# Valeurs par défaut
DEFAULT_FEATURE_COUNT = 78
DEFAULT_MIN_FEATURES = 20
DEFAULT_MAX_FEATURES = 100
DEFAULT_STEP_SIZE = 10
DEFAULT_CV_FOLDS = 3

def load_config():
    """
    Charge la configuration depuis le fichier JSON.
    Si le fichier n'existe pas, retourne les valeurs par défaut.
    
    Returns:
        dict: Configuration des caractéristiques
    """
    if os.path.exists(CONFIG_FILE):
        try:
            with open(CONFIG_FILE, 'r') as f:
                config = json.load(f)
            
            logger.info(f"Configuration des caractéristiques chargée depuis {CONFIG_FILE}")
            return config
        except Exception as e:
            logger.error(f"Erreur lors du chargement de la configuration: {str(e)}")
    
    # Retourner les valeurs par défaut si le fichier n'existe pas ou en cas d'erreur
    logger.info("Utilisation des valeurs par défaut pour la configuration des caractéristiques")
    return {
        "optimal_feature_count": DEFAULT_FEATURE_COUNT,
        "last_updated": None,
        "parameters": {
            "min_features": DEFAULT_MIN_FEATURES,
            "max_features": DEFAULT_MAX_FEATURES,
            "step_size": DEFAULT_STEP_SIZE,
            "cv_folds": DEFAULT_CV_FOLDS
        }
    }

def save_config(config):
    """
    Sauvegarde la configuration dans le fichier JSON.
    
    Args:
        config (dict): Configuration à sauvegarder
    """
    # S'assurer que le répertoire existe
    os.makedirs(os.path.dirname(CONFIG_FILE), exist_ok=True)
    
    # Ajouter un timestamp de mise à jour
    config["last_updated"] = datetime.now().isoformat()
    
    try:
        with open(CONFIG_FILE, 'w') as f:
            json.dump(config, f, indent=2)
        logger.info(f"Configuration des caractéristiques sauvegardée dans {CONFIG_FILE}")
    except Exception as e:
        logger.error(f"Erreur lors de la sauvegarde de la configuration: {str(e)}")

def update_optimal_feature_count(new_count):
    """
    Met à jour le nombre optimal de caractéristiques dans la configuration.
    
    Args:
        new_count (int): Nouveau nombre optimal de caractéristiques
    """
    config = load_config()
    config["optimal_feature_count"] = new_count
    save_config(config)
    logger.info(f"Nombre optimal de caractéristiques mis à jour: {new_count}")

def get_optimal_feature_count():
    """
    Récupère le nombre optimal de caractéristiques depuis la configuration.
    
    Returns:
        int: Nombre optimal de caractéristiques
    """
    config = load_config()
    return config.get("optimal_feature_count", DEFAULT_FEATURE_COUNT)
