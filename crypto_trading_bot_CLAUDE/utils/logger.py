# utils/logger.py
"""
Configuration du système de journalisation
"""
import os
import logging
from logging.handlers import RotatingFileHandler
from typing import Optional

from config.config import LOG_LEVEL, LOG_FORMAT, LOG_FILE, LOG_DIR

# Créer le répertoire de logs s'il n'existe pas
if not os.path.exists(LOG_DIR):
    os.makedirs(LOG_DIR)

def setup_logger(name: str, level: Optional[int] = None, 
                log_format: Optional[str] = None, log_file: Optional[str] = None) -> logging.Logger:
    """
    Configure un logger avec rotation des fichiers
    
    Args:
        name: Nom du logger
        level: Niveau de journalisation
        log_format: Format des messages de log
        log_file: Chemin du fichier de log
    
    Returns:
        Logger configuré
    """
    # Utiliser les valeurs par défaut si non spécifiées
    if level is None:
        level = LOG_LEVEL
    if log_format is None:
        log_format = LOG_FORMAT
    if log_file is None:
        log_file = LOG_FILE.replace('.log', f'_{name}.log')
    
    # Créer et configurer le logger
    logger = logging.getLogger(name)
    
    # Éviter d'ajouter des handlers multiples
    if not logger.handlers:
        logger.setLevel(level)
        
        # Formatter pour les messages de log
        formatter = logging.Formatter(log_format)
        
        # Handler pour la console
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
        
        # Handler pour le fichier avec rotation
        file_handler = RotatingFileHandler(
            log_file, 
            maxBytes=5*1024*1024,  # 5 MB
            backupCount=10
        )
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger
