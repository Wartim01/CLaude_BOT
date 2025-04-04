"""
Utilitaires pour la gestion uniforme des processus externes via subprocess.
Fournit des fonctions cohérentes pour exécuter des commandes en console
avec une gestion d'erreur standardisée.
"""

import subprocess
import logging
import re
from typing import List, Dict, Optional, Tuple, Any

logger = logging.getLogger(__name__)

def run_process(command: List[str], 
               process_name: str, 
               log_output: bool = True,
               capture_output: bool = True,
               working_dir: Optional[str] = None,
               env: Optional[Dict[str, str]] = None,
               timeout: Optional[int] = None,
               expected_return_codes: List[int] = [0]) -> subprocess.CompletedProcess:
    """
    Exécute un processus externe avec une gestion d'erreur standardisée.
    
    Args:
        command: Liste des arguments de la commande
        process_name: Nom descriptif du processus (pour les logs)
        log_output: Si True, enregistre la sortie dans les logs
        capture_output: Si True, capture stdout et stderr
        working_dir: Répertoire de travail pour le processus
        env: Variables d'environnement pour le processus
        timeout: Temps maximum d'exécution en secondes
        expected_return_codes: Liste des codes de retour considérés comme succès
        
    Returns:
        L'objet CompletedProcess avec les résultats
        
    Raises:
        subprocess.CalledProcessError: Si le processus échoue avec un code non attendu
        subprocess.TimeoutExpired: Si le processus dépasse le timeout
    """
    logger.info(f"Démarrage du processus: {process_name}")
    logger.debug(f"Commande: {' '.join(command)}")
    
    try:
        # Exécution du processus avec check=True pour lever une exception en cas d'échec
        result = subprocess.run(
            command,
            capture_output=capture_output,
            text=True,
            check=True,  # Toujours utiliser check=True pour une gestion cohérente
            cwd=working_dir,
            env=env,
            timeout=timeout
        )
        
        # Enregistrement de la sortie si demandé
        if log_output and capture_output:
            if result.stdout:
                logger.debug(f"Sortie standard de {process_name}:\n{result.stdout}")
            if result.stderr:
                logger.debug(f"Erreur standard de {process_name}:\n{result.stderr}")
        
        # Vérification des codes de retour attendus (même si check=True)
        if result.returncode not in expected_return_codes:
            logger.warning(f"Le processus {process_name} s'est terminé avec un code inattendu: {result.returncode}")
        else:
            logger.info(f"Processus {process_name} terminé avec succès (code {result.returncode})")
        
        return result
        
    except subprocess.CalledProcessError as e:
        logger.error(f"Échec du processus {process_name} avec le code {e.returncode}")
        if e.stdout:
            logger.error(f"Sortie standard:\n{e.stdout}")
        if e.stderr:
            logger.error(f"Erreur standard:\n{e.stderr}")
        raise
        
    except subprocess.TimeoutExpired as e:
        logger.error(f"Timeout lors de l'exécution du processus {process_name} après {timeout} secondes")
        if hasattr(e, 'stdout') and e.stdout:
            logger.error(f"Sortie partielle:\n{e.stdout}")
        raise
        
    except Exception as e:
        logger.error(f"Erreur inattendue lors de l'exécution du processus {process_name}: {str(e)}")
        raise

def extract_metric_from_output(output: str, pattern: str, default_value: Any = None) -> Any:
    """
    Extrait une métrique spécifique de la sortie du processus en utilisant une expression régulière.
    
    Args:
        output: Texte de sortie à analyser
        pattern: Expression régulière avec un groupe de capture pour la valeur
        default_value: Valeur à retourner si le pattern n'est pas trouvé
        
    Returns:
        La valeur extraite ou default_value si non trouvée
    """
    match = re.search(pattern, output)
    if match:
        return match.group(1)
    return default_value

def extract_json_metrics(output: str) -> Optional[Dict]:
    """
    Tente d'extraire des métriques au format JSON de la sortie du processus.
    Recherche spécifiquement le format standardisé 'METRICS_SUMMARY|' suivi de paires clé:valeur.
    
    Args:
        output: Texte de sortie à analyser
        
    Returns:
        Dictionnaire de métriques ou None si aucune métrique trouvée
    """
    # Chercher d'abord le format standardisé METRICS_SUMMARY
    metrics_match = re.search(r'METRICS_SUMMARY\|(.*)', output)
    if metrics_match:
        metrics_string = metrics_match.group(1)
        # Convertir les paires clé:valeur en dictionnaire
        try:
            metrics_dict = {}
            for pair in metrics_string.split('|'):
                if ':' in pair:
                    key, value = pair.split(':', 1)
                    try:
                        # Tenter de convertir en float si possible
                        metrics_dict[key] = float(value)
                    except ValueError:
                        metrics_dict[key] = value
            return metrics_dict
        except Exception as e:
            logger.warning(f"Erreur lors de l'extraction des métriques: {str(e)}")
    
    # Chercher ensuite le format FINAL_F1_SCORE
    f1_match = re.search(r'FINAL_F1_SCORE: ([0-9.]+)', output)
    if f1_match:
        try:
            return {"F1": float(f1_match.group(1))}
        except ValueError:
            pass
    
    return None
