"""
Utilitaires pour la gestion des hyperparamètres dans le processus d'optimisation
et d'entraînement des modèles.
"""

import logging
import json
import os
from typing import Dict, List, Any, Optional

logger = logging.getLogger(__name__)

def transform_optimized_params(params: Dict[str, Any]) -> List[str]:
    """
    Transforme un dictionnaire de paramètres optimisés en arguments de ligne de commande
    pour le script train_model.py.
    
    Args:
        params: Dictionnaire contenant les hyperparamètres optimisés
        
    Returns:
        Liste d'arguments de ligne de commande au format ["--param1", "value1", ...]
    """
    cmd_args = []
    
    # Mapping entre les noms de paramètres dans l'optimisation et les arguments en ligne de commande
    param_mapping = {
        # Structure des LSTM
        "lstm_units_first": None,  # Traitement spécial pour lstm_units
        "lstm_layers": None,       # Utilisé avec lstm_units_first pour construire lstm_units
        "lstm_units": "--lstm_units",
        
        # Hyperparamètres d'entraînement
        "dropout_rate": "--dropout",
        "learning_rate": "--learning_rate",
        "batch_size": "--batch_size",
        "sequence_length": "--sequence_length",
        "use_attention": "--use_attention",
        "optimizer": "--optimizer",
        
        # Régularisation
        "l1_regularization": "--l1_reg",
        "l2_regularization": "--l2_reg",
        "l1_reg": "--l1_reg",
        "l2_reg": "--l2_reg",
        
        # Paramètres de séparation des données
        "validation_split": "--validation_split",
        "test_size": "--test_size",
        "train_val_split_method": "--train_val_split_method",
        
        # Autres paramètres
        "feature_scaler": "--feature_scaler",
        "epochs": "--epochs",
    }
    
    # Traitement spécial pour lstm_units
    if "lstm_units_first" in params and "lstm_layers" in params:
        lstm_units_first = params["lstm_units_first"]
        lstm_layers = params["lstm_layers"]
        
        # Construire la liste des unités LSTM (décroissante)
        lstm_units = [lstm_units_first]
        for i in range(1, lstm_layers):
            # Diminuer progressivement le nombre d'unités
            lstm_units.append(lstm_units[-1] // 2)
        
        # Convertir en chaîne de caractères séparée par des virgules
        cmd_args.extend(["--lstm_units", ",".join(map(str, lstm_units))])
        logger.debug(f"Constructed LSTM units: {lstm_units}")
    elif "lstm_units" in params:
        # Si les unités LSTM sont déjà au format liste ou chaîne
        lstm_units = params["lstm_units"]
        if isinstance(lstm_units, list):
            cmd_args.extend(["--lstm_units", ",".join(map(str, lstm_units))])
        else:
            cmd_args.extend(["--lstm_units", str(lstm_units)])
    
    # Traitement spécial pour les paramètres booléens
    bool_params = ["use_attention", "use_early_stopping"]
    
    # Traiter les autres paramètres
    for param_name, cmd_arg in param_mapping.items():
        # Skip special cases handled above
        if param_name in ["lstm_units_first", "lstm_layers", "lstm_units"]:
            continue
            
        if param_name in params and cmd_arg is not None:
            value = params[param_name]
            
            # Traitement des booléens
            if param_name in bool_params:
                if value:  # Si True, ajouter l'argument sans valeur
                    cmd_args.append(cmd_arg)
                # Si False, ne pas ajouter l'argument
            else:
                cmd_args.extend([cmd_arg, str(value)])
    
    # Assurer que les paramètres de régularisation sont inclus (valeurs par défaut si non spécifiés)
    if "--l1_reg" not in cmd_args and "--l2_reg" not in cmd_args:
        # Vérifier l1_regularization et l2_regularization
        if "l1_regularization" in params or "l1_reg" in params:
            l1_val = params.get("l1_regularization", params.get("l1_reg", 0.0001))
            cmd_args.extend(["--l1_reg", str(l1_val)])
        
        if "l2_regularization" in params or "l2_reg" in params:
            l2_val = params.get("l2_regularization", params.get("l2_reg", 0.0001))
            cmd_args.extend(["--l2_reg", str(l2_val)])
    
    logger.debug(f"Transformed parameters to command args: {cmd_args}")
    return cmd_args

def load_optimized_parameters(symbol: str, timeframe: str, model_type: str = "lstm") -> Optional[Dict[str, Any]]:
    """
    Charge les hyperparamètres optimisés pour un symbole et timeframe spécifiques.
    
    Args:
        symbol: Symbole de la paire (ex: 'BTCUSDT')
        timeframe: Intervalle de temps (ex: '1h', '15m')
        model_type: Type de modèle (ex: 'lstm', 'cnn')
        
    Returns:
        Dictionnaire des hyperparamètres optimisés ou None si non trouvé
    """
    # Chemins potentiels pour les fichiers de paramètres optimisés
    paths = [
        f"config/optimized_params_{symbol}_{timeframe}_{model_type}.json",
        f"config/optimized_params_{timeframe}_{model_type}.json",
        f"config/optimized_params_{model_type}.json",
        "config/optimized_params.json"
    ]
    
    # Essayer chaque chemin dans l'ordre
    for path in paths:
        if os.path.exists(path):
            try:
                with open(path, 'r') as f:
                    params = json.load(f)
                
                # Vérifier si les paramètres pour ce symbole/timeframe existent dans le fichier
                if symbol and timeframe:
                    key = f"{symbol}_{timeframe}"
                    if key in params:
                        logger.info(f"Found optimized parameters for {key} in {path}")
                        return params[key]
                
                # Vérifier si les paramètres pour ce timeframe existent
                if timeframe and timeframe in params:
                    logger.info(f"Found optimized parameters for timeframe {timeframe} in {path}")
                    return params[timeframe]
                
                # Utiliser les paramètres génériques
                if model_type in params:
                    logger.info(f"Found generic optimized parameters for {model_type} in {path}")
                    return params[model_type]
                
                # Retourner tout le fichier si aucune correspondance spécifique
                if isinstance(params, dict) and len(params) > 0:
                    logger.info(f"Using full parameter file {path}")
                    return params
                
            except Exception as e:
                logger.warning(f"Error loading optimized parameters from {path}: {str(e)}")
    
    logger.warning(f"No optimized parameters found for {symbol} {timeframe} {model_type}")
    return None

def save_optimized_parameters(params: Dict[str, Any], symbol: str, timeframe: str, model_type: str = "lstm") -> bool:
    """
    Sauvegarde les hyperparamètres optimisés pour un symbole et timeframe spécifiques.
    
    Args:
        params: Dictionnaire des hyperparamètres à sauvegarder
        symbol: Symbole de la paire (ex: 'BTCUSDT')
        timeframe: Intervalle de temps (ex: '1h', '15m')
        model_type: Type de modèle (ex: 'lstm', 'cnn')
        
    Returns:
        True si sauvegarde réussie, False sinon
    """
    # Création du répertoire config si nécessaire
    os.makedirs("config", exist_ok=True)
    
    # Chemin du fichier de paramètres
    path = "config/optimized_params.json"
    
    # Clé pour les paramètres de ce symbole/timeframe
    key = f"{symbol}_{timeframe}"
    
    try:
        # Charger les paramètres existants si le fichier existe
        existing_params = {}
        if os.path.exists(path):
            with open(path, 'r') as f:
                existing_params = json.load(f)
        
        # Mise à jour des paramètres
        existing_params[key] = params
        
        # Ajouter informations de méta-données
        from datetime import datetime
        params["last_optimized"] = datetime.now().isoformat()
        params["symbol"] = symbol
        params["timeframe"] = timeframe
        params["model_type"] = model_type
        
        # Sauvegarde des paramètres mis à jour
        with open(path, 'w') as f:
            json.dump(existing_params, f, indent=2)
            
        logger.info(f"Saved optimized parameters for {key} to {path}")
        return True
        
    except Exception as e:
        logger.error(f"Error saving optimized parameters: {str(e)}")
        return False
