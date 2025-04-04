"""
Module utilitaire pour la gestion des chemins de fichiers
Permet d'accéder aux différents répertoires du projet de manière portable
"""

import os
import pathlib
import logging
from typing import Dict, Optional

# Configuration du logger
logger = logging.getLogger(__name__)

# Déterminer le chemin racine du projet
# Remonte de deux niveaux depuis ce fichier (utils -> racine)
PROJECT_ROOT = pathlib.Path(__file__).parent.parent.resolve()

# Définir les chemins des répertoires principaux
PATHS = {
    "root": PROJECT_ROOT,
    "data": PROJECT_ROOT / "data",
    "models": PROJECT_ROOT / "models",
    "logs": PROJECT_ROOT / "logs",
    "config": PROJECT_ROOT / "config",
    "market_data": PROJECT_ROOT / "data" / "market_data",
    "backtest_results": PROJECT_ROOT / "data" / "backtest_results",
    "trained_models": PROJECT_ROOT / "models" / "trained",
    "scalers": PROJECT_ROOT / "models" / "scalers",
    "optimization": PROJECT_ROOT / "models" / "optimization",
}

# Créer les répertoires s'ils n'existent pas
for path_name, path in PATHS.items():
    os.makedirs(path, exist_ok=True)
    logger.debug(f"Chemin {path_name}: {path}")

def get_project_root() -> pathlib.Path:
    """Retourne le chemin racine du projet"""
    return PROJECT_ROOT

def get_path(path_name: str) -> pathlib.Path:
    """
    Retourne le chemin correspondant à un répertoire spécifique du projet
    
    Args:
        path_name: Nom du chemin à récupérer (ex: 'data', 'models', etc.)
        
    Returns:
        Le chemin absolu correspondant
    """
    if path_name not in PATHS:
        logger.warning(f"Chemin '{path_name}' non défini. Retour au chemin racine.")
        return PROJECT_ROOT
    return PATHS[path_name]

def build_path(*parts: str, base: Optional[str] = None) -> str:
    """
    Construit un chemin en joignant les parties à une base
    
    Args:
        *parts: Parties du chemin à joindre
        base: Nom du répertoire de base (ex: 'data', 'models')
            Si None, utilise le chemin racine du projet
            
    Returns:
        Le chemin absolu construit
    """
    if base is not None:
        base_path = get_path(base)
    else:
        base_path = PROJECT_ROOT
        
    return os.path.join(base_path, *parts)

def get_market_data_path(symbol: str, timeframe: str) -> str:
    """
    Construit le chemin pour les données de marché d'un symbole et timeframe spécifiques
    
    Args:
        symbol: Symbole de la paire (ex: 'BTCUSDT')
        timeframe: Intervalle de temps (ex: '1h', '15m')
        
    Returns:
        Chemin absolu vers le fichier de données
    """
    filename = f"{symbol}_{timeframe}.csv"
    return build_path(filename, base="market_data")

def get_model_path(symbol: str, timeframe: str, version: Optional[str] = None) -> str:
    """
    Construit le chemin pour un modèle entraîné
    
    Args:
        symbol: Symbole de la paire (ex: 'BTCUSDT')
        timeframe: Intervalle de temps (ex: '1h', '15m')
        version: Version du modèle (ex: 'v1', 'baseline')
            Si None, utilise la date actuelle
            
    Returns:
        Chemin absolu vers le fichier modèle
    """
    import datetime
    version = version or datetime.datetime.now().strftime("%Y%m%d_%H%M")
    filename = f"{symbol}_{timeframe}_{version}.keras"
    return build_path(filename, base="trained_models")

def get_scaler_path(symbol: str, timeframe: str, feature_group: str = "lstm") -> str:
    """
    Construit le chemin pour un scaler de caractéristiques
    
    Args:
        symbol: Symbole de la paire
        timeframe: Intervalle de temps
        feature_group: Groupe de features (ex: 'lstm', 'cnn')
        
    Returns:
        Chemin absolu vers le fichier scaler
    """
    filename = f"{symbol}_{timeframe}_{feature_group}_scaler.pkl"
    return build_path(filename, base="scalers")

if __name__ == "__main__":
    # Configurer le logging pour les tests
    logging.basicConfig(level=logging.INFO)
    
    # Afficher les chemins disponibles
    logger.info("Chemins du projet:")
    for name, path in PATHS.items():
        logger.info(f"  {name}: {path}")
        
    # Exemple d'utilisation
    logger.info(f"Chemin des données de marché pour BTCUSDT 1h: {get_market_data_path('BTCUSDT', '1h')}")
    logger.info(f"Chemin du modèle pour ETHUSDT 15m: {get_model_path('ETHUSDT', '15m')}")
