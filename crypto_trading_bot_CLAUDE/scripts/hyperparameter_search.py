# utils/hyperparameter_utils.py

import json
import os
import logging

logger = logging.getLogger(__name__)

def transform_optimized_params(params):
    """
    Transforme un dictionnaire de paramètres optimisés en arguments de ligne de commande.
    
    Args:
        params: Dictionnaire de paramètres optimisés
        
    Returns:
        Liste d'arguments de ligne de commande
    """
    cmd_args = []
    for key, value in params.items():
        if isinstance(value, bool):
            if value:
                cmd_args.append(f"--{key}")
        else:
            cmd_args.append(f"--{key}")
            cmd_args.append(str(value))
    return cmd_args

def save_optimized_parameters(params, symbol, timeframe):
    """
    Sauvegarde les paramètres optimisés dans un fichier JSON.
    
    Args:
        params: Dictionnaire de paramètres optimisés
        symbol: Symbole de la paire de trading
        timeframe: Intervalle de temps
    """
    # Chemin du fichier de sauvegarde
    from utils.path_utils import build_path
    save_path = build_path(f"best_params_{symbol}_{timeframe}.json", base="optimization")
    
    try:
        with open(save_path, 'w') as f:
            json.dump(params, f, indent=2)
        logger.info(f"Paramètres optimisés sauvegardés dans {save_path}")
    except Exception as e:
        logger.error(f"Erreur lors de la sauvegarde des paramètres optimisés: {str(e)}")