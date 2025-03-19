import subprocess
import sys
import os
import logging
from utils.logger import setup_logger
from config.config import DATA_DIR
# Ajout d'import optionnels si nécessaire (par exemple pour lire des arguments ou traiter des données)
import argparse
import pandas as pd

# Configuration d'un logger dédié à la pipeline
pipeline_log_path = os.path.join(os.path.dirname(__file__), "pipeline.log")
pipeline_logger = logging.getLogger("pipeline")
pipeline_logger.setLevel(logging.DEBUG)
if not pipeline_logger.handlers:
    file_handler = logging.FileHandler(pipeline_log_path)
    file_handler.setLevel(logging.DEBUG)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    pipeline_logger.addHandler(file_handler)

# Paramètres communs pour train_model.py et hyperparameter_search.py
common_args = {
    "--symbol": "BTCUSDT",
    "--timeframe": "15m",
    "--start-date": "2022-01-01",
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
    pipeline_logger.info("Début du téléchargement des données...")
    cmd = [
        sys.executable,
        "download_data.py",
    ] + build_download_args()
    pipeline_logger.debug("Commande de téléchargement: " + " ".join(cmd))
    result = subprocess.run(cmd, capture_output=True, text=True)
    pipeline_logger.info("Sortie stdout du téléchargement: " + result.stdout)
    if result.stderr:
        pipeline_logger.error("Sortie stderr du téléchargement: " + result.stderr)
    pipeline_logger.info("Téléchargement terminé.")

def run_hyperparameter_search():
    pipeline_logger.info("Début de la recherche d'hyperparamètres...")
    cmd = [
        sys.executable,
        "hyperparameter_search.py",
    ] + build_common_args()
    pipeline_logger.debug("Commande de recherche d'hyperparamètres: " + " ".join(cmd))
    result = subprocess.run(cmd, capture_output=True, text=True)
    pipeline_logger.info("Sortie stdout de la recherche d'hyperparamètres: " + result.stdout)
    if result.stderr:
        pipeline_logger.error("Sortie stderr de la recherche d'hyperparamètres: " + result.stderr)
    pipeline_logger.info("Recherche d'hyperparamètres terminée.")

def run_training():
    pipeline_logger.info("Début de l'entraînement avec les hyperparamètres optimisés...")
    cmd = [
        sys.executable,
        "train_model.py",
        "train",
    ] + build_common_args()
    pipeline_logger.debug("Commande d'entraînement: " + " ".join(cmd))
    result = subprocess.run(cmd, capture_output=True, text=True)
    pipeline_logger.info("Sortie stdout de l'entraînement: " + result.stdout)
    if result.stderr:
        pipeline_logger.error("Sortie stderr de l'entraînement: " + result.stderr)
    pipeline_logger.info("Entraînement terminé.")

if __name__ == "__main__":
    pipeline_logger.info("Début de l'exécution du pipeline.")
    run_download_data()
    run_hyperparameter_search()
    run_training()
    pipeline_logger.info("Exécution du pipeline terminée.")
