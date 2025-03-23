import subprocess
import sys
import os
import logging
import argparse
import pandas as pd

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

# Common argument building function
common_args = {
    "--symbol": "BTCUSDT",
    "--timeframe": "15m",
    "--start-date": "2024-01-01",
    "--end-date": "2024-12-31",
    "--data_path": r"C:\Users\timot\OneDrive\Bureau\BOT TRADING BIG 2025\crypto_trading_bot_CLAUDE\data\market_data"
}

def build_common_args():
    """Retourne la liste d'arguments communs pour les scripts utilisant --start-date / --end-date."""
    args_list = []
    for key, value in common_args.items():
        args_list.extend([key, value])
    return args_list

def build_download_args():
    """Retourne la liste d'arguments pour download_data.py qui utilise --start et --end."""
    # On réutilise les mêmes valeurs que pour common_args, mais avec les clés attendues par download_data.py
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

def run_download_data():
    try:
        pipeline_logger.info("Début du téléchargement des données...")
        cmd = [sys.executable, "download_data.py"] + build_download_args()
        pipeline_logger.debug("Commande de téléchargement: " + " ".join(cmd))
        
        # Set environment to ensure UTF-8 encoding
        env = os.environ.copy()
        env["PYTHONIOENCODING"] = "utf-8"
        
        # Use explicit UTF-8 encoding for subprocess
        result = subprocess.run(cmd, capture_output=True, text=True, encoding='utf-8', env=env, check=True)
        pipeline_logger.info("Sortie stdout du téléchargement: " + result.stdout)
        pipeline_logger.info("Téléchargement terminé.")
        return True
    except subprocess.CalledProcessError as e:
        pipeline_logger.error(f"Erreur lors du téléchargement des données: {e}")
        pipeline_logger.error(f"Sortie stderr: {e.stderr}")
        return False

def run_hyperparameter_search():
    try:
        pipeline_logger.info("Début de la recherche d'hyperparamètres...")
        cmd = [sys.executable, "hyperparameter_search.py"] + build_common_args()
        pipeline_logger.debug("Commande de recherche d'hyperparamètres: " + " ".join(cmd))
        
        # Set environment to ensure UTF-8 encoding
        env = os.environ.copy()
        env["PYTHONIOENCODING"] = "utf-8"
        
        result = subprocess.run(cmd, capture_output=True, text=True, encoding='utf-8', env=env, check=True)
        pipeline_logger.info("Sortie stdout de la recherche d'hyperparamètres: " + result.stdout)
        pipeline_logger.info("Recherche d'hyperparamètres terminée.")
    except Exception as e:
        pipeline_logger.error(f"Erreur lors de la recherche d'hyperparamètres: {e}")

def run_training():
    try:
        pipeline_logger.info("Début de l'entraînement avec les hyperparamètres optimisés...")
        # Add the 'train' subcommand now as expected by train_model.py
        # Ajouter le paramètre --verbose=2 pour voir les métriques détaillées
        cmd = [sys.executable, "train_model.py", "train"] + build_common_args() + ["--verbose", "2"]
        pipeline_logger.debug("Commande d'entraînement: " + " ".join(cmd))
        
        # Set environment to ensure UTF-8 encoding
        env = os.environ.copy()
        env["PYTHONIOENCODING"] = "utf-8"
        
        # Execute with full error capture and proper encoding
        # Utiliser subprocess.run() au lieu de Popen pour afficher la sortie standard en temps réel
        result = subprocess.run(
            cmd,
            check=False,  # Ne pas lever d'exception si le code de retour est non-zéro
            text=True,
            encoding='utf-8',
            errors='replace',  # Add this parameter to handle Unicode decoding errors
            env=env
        )
        
        if result.returncode != 0:
            pipeline_logger.error(f"Erreur lors de l'entraînement (code {result.returncode})")
            return False
            
        pipeline_logger.info("Entraînement terminé.")
        return True
    except Exception as e:
        pipeline_logger.error(f"Erreur lors de l'entraînement: {e}")
        return False

def run_evaluation():
    try:
        pipeline_logger.info("Début de l'évaluation du modèle...")
        # Add the 'evaluate' subcommand as attendu by evaluate_model.py
        cmd = [sys.executable, "evaluate_model.py", "evaluate"] + build_common_args()
        pipeline_logger.debug("Commande d'évaluation: " + " ".join(cmd))
        
        # Set environment to ensure UTF-8 encoding
        env = os.environ.copy()
        env["PYTHONIOENCODING"] = "utf-8"
        
        # Use subprocess.run instead of Popen for simpler error handling and real-time output
        result = subprocess.run(
            cmd,
            check=False,  # Don't raise exception on non-zero exit codes
            text=True,    # Automatically decode output
            encoding='utf-8',
            errors='replace',  # Replace invalid UTF-8 characters instead of failing
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
    parser.add_argument("--run-hp-search", action="store_true", help="Lancer la recherche d'hyperparamètres")
    parser.add_argument("--model-type", type=str, default="lstm", help="Type de modèle (lstm, transformer, etc.)")
    parser.add_argument("--data-path", type=str, default=os.path.join(DATA_DIR, "market_data"), help="Chemin des données de marché")
    parser.add_argument("--skip-download", action="store_true", help="Sauter l'étape de téléchargement des données")
    parser.add_argument("--skip-training", action="store_true", help="Sauter l'étape d'entraînement")
    parser.add_argument("--skip-evaluation", action="store_true", help="Sauter l'étape d'évaluation")
    parser.add_argument("--hyperparameter-search", action="store_true",
                        help="Lance la recherche d'hyperparamètres avant l'entraînement")

    args = parser.parse_args()

    if args.hyperparameter_search:
        pipeline_logger.info("Lancement de la recherche d'hyperparamètres...")
        run_hyperparameter_search()

    pipeline_logger.info("Début de l'exécution du pipeline.")
    success = True
    
    # Step 1: Download data if not skipped
    if not args.skip_download:
        if not run_download_data():
            pipeline_logger.warning("Problème lors du téléchargement des données, mais continue le pipeline...")
            success = False
    else:
        pipeline_logger.info("Étape de téléchargement des données sautée.")
    
    # Step 2: Run hyperparameter search if requested
    if args.run_hp_search:
        try:
            from hyperparameter_search import main as run_hp_search
            pipeline_logger.info("Début de la recherche d'hyperparamètres...")
            run_hp_search()
            pipeline_logger.info("Recherche d'hyperparamètres terminée.")
        except Exception as e:
            pipeline_logger.error(f"Erreur lors de la recherche d'hyperparamètres: {e}")
            success = False
    
    # Step 3: Run training if not skipped
    if not args.skip_training:
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
