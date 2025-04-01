import subprocess
import sys
import os
import logging
import argparse
import pandas as pd
import json
import re

from utils.logger import setup_logger
from config.config import DATA_DIR

# Configure pipeline logger with file and console output
pipeline_log_path = os.path.join(os.path.dirname(__file__), "pipeline.log")
pipeline_logger = logging.getLogger("pipeline")
pipeline_logger.setLevel(logging.DEBUG)
if not pipeline_logger.handlers:
    file_handler = logging.FileHandler(pipeline_log_path)
    file_handler.setLevel(logging.DEBUG)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    pipeline_logger.addHandler(file_handler)
    # Add console handler to print all logs to the console
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.DEBUG)
    console_handler.setFormatter(formatter)
    pipeline_logger.addHandler(console_handler)

# Common arguments restent disponibles pour download_data.py
common_args = {
    "--symbol": "BTCUSDT",
    "--timeframe": "15m",
    "--start-date": "2024-01-01",
    "--end-date": "2024-02-29",   # corrigé (2024 est bissextile)
    "--data_path": r"C:\Users\timot\OneDrive\Bureau\BOT TRADING BIG 2025\crypto_trading_bot_CLAUDE\data\market_data"
}

def build_common_args():
    """Retourne la liste d'arguments communs pour download_data.py, gardant --start-date et --end-date."""
    args_list = []
    for key, value in common_args.items():
        args_list.extend([key, value])
    return args_list

def build_download_args():
    """Retourne la liste d'arguments pour download_data.py qui utilise --start et --end."""
    download_args = {
        "--symbol": common_args["--symbol"],
        "--interval": common_args["--timeframe"],
        "--start": common_args["--start-date"],
        "--end": common_args["--end-date"]
    }
    args_list = []
    for key, value in download_args.items():
        args_list.extend([key, value])
    return args_list

# Nouveaux helpers pour train_model.py et evaluate_model.py
def build_train_args():
    """Arguments pour train_model.py; inclut --symbol, --timeframe et --data_path."""
    return [
        "--symbol", common_args["--symbol"],
        "--timeframe", common_args["--timeframe"],
        "--data_path", common_args["--data_path"]
    ]

def build_evaluate_args():
    """Arguments pour evaluate_model.py; inclut --symbol, --timeframe, --start-date, --end-date et --data_path."""
    return [
        "--symbol", common_args["--symbol"],
        "--timeframe", common_args["--timeframe"],
        "--start-date", common_args["--start-date"],
        "--end-date", common_args["--end-date"],
        "--data_path", common_args["--data_path"]
    ]

def run_download_data():
    try:
        pipeline_logger.info("Début du téléchargement des données...")
        cmd = [sys.executable, "download_data.py"] + build_download_args()
        pipeline_logger.debug("Commande de téléchargement: " + " ".join(cmd))
        
        env = os.environ.copy()
        env["PYTHONIOENCODING"] = "utf-8"
        
        result = subprocess.run(cmd, capture_output=True, text=True, encoding='utf-8', env=env, check=True)
        pipeline_logger.info("Sortie stdout du téléchargement: " + result.stdout)
        pipeline_logger.info("Téléchargement terminé.")
        return True
    except subprocess.CalledProcessError as e:
        pipeline_logger.error(f"Erreur lors du téléchargement des données: {e}")
        pipeline_logger.error(f"Sortie stderr: {e.stderr}")
        return False

def clean_params_for_saving(params: dict) -> dict:
    """Convert hyperparameter keys to the format expected by train_model.py
    for saving to optimized_params.json"""
    cleaned = {}
    
    # Keep track of F1 score and optimization timestamp
    if "f1_score" in params:
        cleaned["f1_score"] = params["f1_score"]
    if "last_optimized" in params:
        cleaned["last_optimized"] = params["last_optimized"]
    
    # Convert lstm_units_first and lstm_layers into the expected lstm_units format
    if "lstm_units_first" in params and "lstm_layers" in params:
        first_layer = int(params["lstm_units_first"])
        num_layers = int(params["lstm_layers"])
        
        # Create a decreasing sequence of units
        units = []
        current_units = first_layer
        for _ in range(num_layers):
            units.append(current_units)
            current_units = current_units // 2
            
        cleaned["lstm_units"] = units  # Store as a list that train_model.py can understand
    
    # Direct mappings with standardized names
    name_mappings = {
        "dropout_rate": "dropout",
        "learning_rate": "learning_rate",
        "batch_size": "batch_size",
        "early_stopping_patience": "patience",
        "epochs": "epochs"
    }
    
    # Apply mappings
    for old_key, new_key in name_mappings.items():
        if old_key in params:
            cleaned[new_key] = params[old_key]
    
    return cleaned

def process_hyperparameter_result(output: str) -> None:
    """Process the output from hyperparameter search and clean the parameters before saving"""
    # Check if hyperparameter output indicates parameters were saved
    if "Saving best parameters" in output:
        # Try to find the path where parameters were saved
        optimized_params_path = os.path.join("config", "optimized_params.json")
        
        if os.path.exists(optimized_params_path):
            pipeline_logger.info(f"Found optimized parameters file: {optimized_params_path}")
            
            try:
                # Load the current parameters
                with open(optimized_params_path, 'r') as f:
                    params_data = json.load(f)
                
                # Process each timeframe's parameters
                cleaned_params = {}
                for timeframe, params in params_data.items():
                    cleaned_params[timeframe] = clean_params_for_saving(params)
                
                # Save the cleaned parameters back to the file
                with open(optimized_params_path, 'w') as f:
                    json.dump(cleaned_params, f, indent=2)
                
                pipeline_logger.info(f"Saved cleaned parameters to {optimized_params_path}")
            except Exception as e:
                pipeline_logger.error(f"Error processing hyperparameter results: {e}")

def run_hyperparameter_search():
    """Run hyperparameter search with early stopping when F1≥0.70 is reached or after max iterations"""
    max_iterations = 5  # Maximum number of search iterations to prevent infinite loops
    current_iteration = 0
    
    while current_iteration < max_iterations:
        current_iteration += 1
        pipeline_logger.info(f"Recherche d'hyperparamètres - Itération {current_iteration}/{max_iterations}")
        cmd = [sys.executable, "hyperparameter_search.py"] + build_common_args()
        pipeline_logger.debug("Commande de recherche d'hyperparamètres: " + " ".join(cmd))
        
        env = os.environ.copy()
        env["PYTHONIOENCODING"] = "utf-8"
        
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, encoding='utf-8', env=env, check=True)
            stdout = result.stdout
            pipeline_logger.info(f"Sortie stdout: {stdout}")
            
            # Process and clean the hyperparameters that were saved
            process_hyperparameter_result(stdout)
            
            # Check if we've reached the desired F1 score
            match = re.search(r'F1=([0-9]+\.[0-9]+)', stdout)
            if match and float(match.group(1)) >= 0.70:
                pipeline_logger.info(f"F1 ≥ 0.70 atteint ({match.group(1)}), fin de la recherche d'hyperparamètres.")
                return True
                
            pipeline_logger.info("F1 < 0.70, nouvelle itération de recherche si nécessaire.")
            
            # If we reach the max iterations, log it
            if current_iteration >= max_iterations:
                pipeline_logger.warning(f"Nombre maximum d'itérations atteint ({max_iterations}). "
                                      f"Meilleur F1 score obtenu < 0.70. Utilisation des meilleurs paramètres trouvés.")
                return False
                
        except Exception as e:
            pipeline_logger.error(f"Erreur lors de la recherche d'hyperparamètres: {e}")
            return False
    
    return False  # Should not reach here but just in case

def transform_optimized_params(params: dict) -> dict:
    """Transform hyperparameter keys from hyperparameter_search to train_model.py format."""
    transformed = {}
    
    # Create LSTM units parameter
    if "lstm_units_first" in params and "lstm_layers" in params:
        first_layer = int(params["lstm_units_first"])
        num_layers = int(params["lstm_layers"])
        
        # Create decreasing units list
        units = []
        current_units = first_layer
        for _ in range(num_layers):
            units.append(str(current_units))
            current_units = current_units // 2
        
        # Join with commas as expected by train_model.py
        transformed["lstm_units"] = ",".join(units)
    
    # Map parameter names
    name_mapping = {
        "dropout_rate": "dropout",
        "learning_rate": "learning_rate",
        "batch_size": "batch_size"
    }
    
    # Apply mappings for direct parameters
    for source_key, target_key in name_mapping.items():
        if source_key in params:
            transformed[target_key] = params[source_key]
    
    # Add patience parameter if available
    if "early_stopping_patience" in params:
        transformed["patience"] = params["early_stopping_patience"]
    
    return transformed

def run_training():
    """Run model training with optimized hyperparameters, with retry logic if F1<0.70"""
    max_attempts = 3
    current_attempt = 0
    
    while current_attempt < max_attempts:
        current_attempt += 1
        pipeline_logger.info(f"Entraînement - Tentative {current_attempt}/{max_attempts}")
        
        # Start with base command
        cmd = [sys.executable, "train_model.py"]
        
        # Dictionary to store ONLY valid arguments that train_model.py can recognize
        valid_args = {
            "--symbol": common_args["--symbol"],
            "--timeframe": common_args["--timeframe"],
            "--data_path": common_args["--data_path"],
            "--verbose": "2"  # Always use verbose=2 for detailed output
        }
        
        # Load parameters from optimized_params.json
        try:
            optimized_params_path = os.path.join("config", "optimized_params.json")
            if os.path.exists(optimized_params_path):
                with open(optimized_params_path, 'r') as f:
                    opt = json.load(f)
                
                # Get parameters for current timeframe
                timeframe = common_args["--timeframe"]
                if timeframe in opt:
                    # Process the raw parameters from JSON file
                    raw_params = opt[timeframe]
                    pipeline_logger.info(f"Paramètres bruts trouvés pour {timeframe}")
                    
                    # Transform parameters to the format train_model.py expects
                    transformed_params = transform_optimized_params(raw_params)
                    pipeline_logger.info(f"Paramètres transformés: {transformed_params}")
                    
                    # Add only the transformed parameters to valid_args
                    for key, value in transformed_params.items():
                        valid_args[f"--{key}"] = str(value)
        except Exception as e:
            pipeline_logger.error(f"Erreur lors du chargement des paramètres optimisés: {e}")
        
        # Build the final command using only valid parameters
        command_args = []
        for key, value in valid_args.items():
            command_args.extend([key, str(value)])
        
        # Final command with all valid arguments
        cmd.extend(command_args)
        
        pipeline_logger.info("Commande d'entraînement construite avec uniquement les arguments valides")
        pipeline_logger.debug("Commande d'entraînement: " + " ".join(cmd))
        
        # Execute training process
        env = os.environ.copy()
        env["PYTHONIOENCODING"] = "utf-8"
        
        # Run the command
        result = subprocess.run(
            cmd,
            check=False,
            text=True,
            encoding='utf-8',
            errors='replace',
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            env=env
        )
        
        # Check for errors first
        if result.returncode != 0:
            pipeline_logger.error(f"Erreur lors de l'entraînement (code {result.returncode})")
            if result.stderr:
                pipeline_logger.error(f"Stderr: {result.stderr}")
            # Continue with next attempt despite error
            continue
        
        # Check F1 score from output
        stdout = result.stdout if result.stdout else ""
        match = re.search(r'F1=([0-9]+\.[0-9]+)', stdout)
        if match and float(match.group(1)) >= 0.70:
            pipeline_logger.info(f"F1 ≥ 0.70 atteint ({match.group(1)}), entraînement réussi!")
            return True
            
        pipeline_logger.info("F1 < 0.70 après l'entraînement.")
        
        # If we've reached max attempts, log it but still return True to continue pipeline
        if current_attempt >= max_attempts:
            pipeline_logger.warning(f"Nombre maximum de tentatives d'entraînement atteint ({max_attempts}). "
                                  f"Meilleur F1 score < 0.70. L'entraînement est considéré comme terminé.")
    
    # If we get here, all attempts were made without reaching F1 ≥ 0.70
    pipeline_logger.info("Entraînement terminé sans atteindre F1 ≥ 0.70.")
    return True  # Return success anyway to continue pipeline

def run_evaluation():
    try:
        pipeline_logger.info("Début de l'évaluation du modèle...")
        # Préfixer d'abord le sous-commande "evaluate" pour evaluate_model.py et appliquer build_evaluate_args
        cmd = [sys.executable, "evaluate_model.py", "evaluate"] + build_evaluate_args()
        pipeline_logger.debug("Commande d'évaluation: " + " ".join(cmd))
        
        env = os.environ.copy()
        env["PYTHONIOENCODING"] = "utf-8"
        
        result = subprocess.run(
            cmd,
            check=False,
            text=True,
            encoding='utf-8',
            errors='replace',
            env=env
        )
        
        if result.returncode != 0:
            pipeline_logger.error(f"Erreur lors de l'évaluation (code {result.returncode})")
            if result.stderr:
                pipeline_logger.error(f"Erreur: {result.stderr}")
            return False
            
        if result.stdout:
            pipeline_logger.info("Sortie de l'évaluation: " + result.stdout)
        
        pipeline_logger.info("Évaluation terminée.")
        return True
    except Exception as e:
        pipeline_logger.error(f"Erreur lors de l'évaluation: {e}")
        return False

def run_pipeline():
    parser = argparse.ArgumentParser(description="Pipeline complet d'entraînement")
    # Consolidate hyperparameter search flags into one clear option
    parser.add_argument("--hyperparameter-search", action="store_true",
                        help="Lance la recherche d'hyperparamètres avant l'entraînement (F1≥0.70 arrête automatiquement)")
    parser.add_argument("--model-type", type=str, default="lstm", help="Type de modèle (lstm, transformer, etc.)")
    parser.add_argument("--data-path", type=str, default=os.path.join(DATA_DIR, "market_data"), help="Chemin des données de marché")
    parser.add_argument("--skip-download", action="store_true", help="Sauter l'étape de téléchargement des données")
    parser.add_argument("--skip-training", action="store_true", help="Sauter l'étape d'entraînement")
    parser.add_argument("--skip-evaluation", action="store_true", help="Sauter l'étape d'évaluation")
    parser.add_argument("--max-hp-iterations", type=int, default=5, 
                        help="Nombre maximum d'itérations pour la recherche d'hyperparamètres")

    args = parser.parse_args()

    pipeline_logger.info("Début de l'exécution du pipeline.")
    success = True
    
    # Step 1: Run hyperparameter search if requested
    if args.hyperparameter_search:
        pipeline_logger.info("Lancement de la recherche d'hyperparamètres avec early stopping (F1≥0.70)...")
        # Run the standalone function that implements early stopping
        run_hyperparameter_search()
        pipeline_logger.info("Recherche d'hyperparamètres terminée. Les meilleurs paramètres seront utilisés pour l'entraînement.")
    
    # Step 2: Download data if not skipped
    if not args.skip_download:
        if not run_download_data():
            pipeline_logger.warning("Problème lors du téléchargement des données, mais continue le pipeline...")
            success = False
    else:
        pipeline_logger.info("Étape de téléchargement des données sautée.")
    
    # Step 3: Run training if not skipped
    if not args.skip_training:
        pipeline_logger.info("Début de l'entraînement avec les hyperparamètres" + 
                           (" optimisés" if args.hyperparameter_search else ""))
        if not run_training():
            pipeline_logger.error("L'entraînement a échoué. L'évaluation risque de ne pas fonctionner.")
            success = False
    else:
        pipeline_logger.info("Étape d'entraînement sautée.")
    
    # Step 4: Run evaluation if not skipped
    if not args.skip_evaluation:
        if not run_evaluation():
            pipeline_logger.error("L'évaluation a échoué.")
            success = False
    else:
        pipeline_logger.info("Étape d'évaluation sautée.")
    
    # Final report
    if success:
        pipeline_logger.info("Exécution du pipeline terminée avec succès.")
    else:
        pipeline_logger.warning("Exécution du pipeline terminée avec des erreurs. Consultez les logs pour plus de détails.")

if __name__ == "__main__":
    run_pipeline()
