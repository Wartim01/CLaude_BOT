"""
Module permettant la séparation des données en ensembles d'entraînement,
validation et test pour les séries temporelles financières.

Supporte différentes méthodes de division:
- Temporelle: respecte l'ordre chronologique (par défaut)
- Aléatoire: mélange les données (moins adapté aux séries temporelles)
- Stratifiée: maintient la distribution de la variable cible
"""

import pandas as pd
import numpy as np
from typing import Tuple, Dict, Optional, Union, List
from sklearn.model_selection import train_test_split, TimeSeriesSplit
import logging

logger = logging.getLogger(__name__)

def split_data(
    data: pd.DataFrame,
    sequence_length: int = 60,
    prediction_horizon: int = 12,
    test_size: float = 0.2,
    validation_size: float = 0.2,
    method: str = 'time',
    random_state: int = 42,
    target_column: Optional[str] = None,
    feature_columns: Optional[List[str]] = None
) -> Dict[str, pd.DataFrame]:
    """
    Divise un DataFrame en ensembles d'entraînement, validation et test
    
    Args:
        data: DataFrame pandas contenant les données
        sequence_length: Longueur des séquences pour les modèles séquentiels
        prediction_horizon: Horizon de prédiction (périodes à prévoir dans le futur)
        test_size: Proportion des données pour l'ensemble de test
        validation_size: Proportion des données d'entraînement pour la validation
        method: Méthode de fractionnement ('time', 'random', 'stratified')
        random_state: Graine aléatoire pour la reproductibilité
        target_column: Colonne cible pour la stratification (si method='stratified')
        feature_columns: Liste des colonnes à utiliser comme caractéristiques
        
    Returns:
        Dictionnaire contenant les DataFrames d'entraînement, validation et test
    """
    # Vérification des paramètres
    if method not in ['time', 'random', 'stratified']:
        logger.warning(f"Méthode de fractionnement '{method}' non reconnue. Utilisation de 'time'")
        method = 'time'
    
    # Vérifier si stratification sans colonne cible
    if method == 'stratified' and target_column is None:
        logger.warning("Méthode stratifiée demandée mais aucune colonne cible spécifiée. Utilisation de 'time'")
        method = 'time'
    
    # Copier les données pour éviter les modifications sur l'original
    data_copy = data.copy()
    
    # Vérifier si les données sont suffisantes
    if len(data_copy) < sequence_length + prediction_horizon + 10:  # +10 pour avoir un minimum raisonnable
        logger.error(f"Données insuffisantes: {len(data_copy)} observations. Minimum recommandé: {sequence_length + prediction_horizon + 10}")
        raise ValueError(f"Données insuffisantes pour créer des séquences de longueur {sequence_length} avec horizon {prediction_horizon}")
    
    # Pour la méthode temporelle (par défaut pour les séries temporelles)
    if method == 'time':
        # Calculer les indices de division
        n_samples = len(data_copy)
        test_idx = int(n_samples * (1 - test_size))
        train_idx = int(test_idx * (1 - validation_size))
        
        # Loguer les informations de division
        logger.info(f"Division temporelle: {train_idx} pour entraînement, {test_idx-train_idx} pour validation, {n_samples-test_idx} pour test")
        
        # Division en maintenant l'ordre temporel
        train_data = data_copy.iloc[:train_idx].copy()
        val_data = data_copy.iloc[train_idx:test_idx].copy()
        test_data = data_copy.iloc[test_idx:].copy()
        
        # Vérifier que les ensembles ne se chevauchent pas pour les modèles séquentiels
        # Une séquence ne devrait pas contenir des données de différents ensembles
        # Pour éviter les fuites de données, nous ajustons les points de coupure
        if train_idx - sequence_length < 0:
            logger.warning(f"Attention: séquences d'entraînement trop longues ({sequence_length}) par rapport aux données disponibles")
        else:
            # Ajuster l'ensemble de validation pour qu'il commence après la dernière séquence d'entraînement
            val_data = data_copy.iloc[(train_idx - sequence_length + 1):test_idx].copy()
            
        # Ajuster l'ensemble de test pour qu'il commence après la dernière séquence de validation
        if test_idx - sequence_length < train_idx:
            logger.warning("Attention: chevauchement possible entre validation et test en raison de la longueur des séquences")
        else:
            test_data = data_copy.iloc[(test_idx - sequence_length + 1):].copy()
    
    # Pour la méthode aléatoire (moins adaptée aux séries temporelles)
    elif method == 'random':
        logger.warning("Division aléatoire utilisée pour les séries temporelles - peut conduire à des fuites de données")
        
        # Première division pour obtenir l'ensemble de test
        train_val_data, test_data = train_test_split(
            data_copy, test_size=test_size, random_state=random_state
        )
        
        # Deuxième division pour séparer entraînement et validation
        train_data, val_data = train_test_split(
            train_val_data, test_size=validation_size, random_state=random_state
        )
    
    # Pour la méthode stratifiée (utile pour les problèmes de classification déséquilibrés)
    elif method == 'stratified':
        logger.info(f"Division stratifiée selon la colonne '{target_column}'")
        
        # Vérifie si la colonne cible existe
        if target_column not in data_copy.columns:
            logger.error(f"Colonne cible '{target_column}' non trouvée dans les données")
            raise ValueError(f"Colonne cible '{target_column}' non trouvée")
        
        # Première division stratifiée pour obtenir l'ensemble de test
        train_val_data, test_data = train_test_split(
            data_copy, test_size=test_size, random_state=random_state,
            stratify=data_copy[target_column]
        )
        
        # Deuxième division stratifiée pour séparer entraînement et validation
        train_data, val_data = train_test_split(
            train_val_data, test_size=validation_size, random_state=random_state,
            stratify=train_val_data[target_column]
        )
    
    # Journalisation des tailles des ensembles
    logger.info(f"Division des données - Entraînement: {len(train_data)}, Validation: {len(val_data)}, Test: {len(test_data)}")
    
    # Vérification des proportions
    total = len(data_copy)
    train_pct = len(train_data) / total * 100
    val_pct = len(val_data) / total * 100
    test_pct = len(test_data) / total * 100
    
    logger.info(f"Proportions - Entraînement: {train_pct:.1f}%, Validation: {val_pct:.1f}%, Test: {test_pct:.1f}%")
    
    # Réindexer les DataFrame pour éviter les problèmes d'index dupliqués
    train_data = train_data.reset_index(drop=True)
    val_data = val_data.reset_index(drop=True)
    test_data = test_data.reset_index(drop=True)
    
    # Retourner un dictionnaire avec les ensembles de données
    return {
        'train': train_data,
        'validation': val_data,
        'test': test_data
    }

def split_sequence_data(
    X: np.ndarray, 
    y: np.ndarray, 
    test_size: float = 0.2, 
    validation_size: float = 0.2,
    method: str = 'time',
    random_state: int = 42
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Divise des données séquentielles (déjà formatées pour les modèles LSTM/RNN)
    en ensembles d'entraînement, validation et test
    
    Args:
        X: Tableau numpy contenant les séquences d'entrée (shape: [n_samples, seq_length, n_features])
        y: Tableau numpy contenant les cibles (shape: [n_samples, ...])
        test_size: Proportion des données pour l'ensemble de test
        validation_size: Proportion des données d'entraînement pour la validation
        method: Méthode de fractionnement ('time', 'random')
        random_state: Graine aléatoire pour la reproductibilité
        
    Returns:
        Tuple contenant X_train, X_val, X_test, y_train, y_val, y_test
    """
    n_samples = X.shape[0]
    
    if method == 'time':
        # Calcul des points de coupure
        test_idx = int(n_samples * (1 - test_size))
        val_idx = int(test_idx * (1 - validation_size))
        
        # Division temporelle
        X_train, y_train = X[:val_idx], y[:val_idx]
        X_val, y_val = X[val_idx:test_idx], y[val_idx:test_idx]
        X_test, y_test = X[test_idx:], y[test_idx:]
        
    elif method == 'random':
        # Génération d'indices aléatoires
        indices = np.random.RandomState(random_state).permutation(n_samples)
        
        # Division avec les indices
        test_split = int(n_samples * test_size)
        val_split = int((n_samples - test_split) * validation_size)
        
        test_indices = indices[:test_split]
        val_indices = indices[test_split:test_split+val_split]
        train_indices = indices[test_split+val_split:]
        
        X_train, y_train = X[train_indices], y[train_indices]
        X_val, y_val = X[val_indices], y[val_indices]
        X_test, y_test = X[test_indices], y[test_indices]
    else:
        raise ValueError(f"Méthode de fractionnement '{method}' non reconnue. Utilisez 'time' ou 'random'")
    
    logger.info(f"Dimensions des données fractionnées:")
    logger.info(f"  X_train: {X_train.shape}, y_train: {y_train.shape}")
    logger.info(f"  X_val: {X_val.shape}, y_val: {y_val.shape}")
    logger.info(f"  X_test: {X_test.shape}, y_test: {y_test.shape}")
    
    return X_train, X_val, X_test, y_train, y_val, y_test

def create_walk_forward_cv(
    data: pd.DataFrame, 
    n_splits: int = 5, 
    test_size: int = 30,
    gap: int = 0
) -> List[Tuple[np.ndarray, np.ndarray]]:
    """
    Crée des indices pour une validation croisée avec marche en avant (walk-forward),
    adaptée aux séries temporelles où l'ordre chronologique est important
    
    Args:
        data: DataFrame contenant les données
        n_splits: Nombre de plis pour la validation croisée
        test_size: Taille de chaque fenêtre de test (en nombre d'observations)
        gap: Nombre d'observations à ignorer entre l'entraînement et le test
        
    Returns:
        Liste de tuples (indices_entrainement, indices_test)
    """
    tscv = TimeSeriesSplit(n_splits=n_splits, test_size=test_size, gap=gap)
    splits = []
    
    for train_idx, test_idx in tscv.split(data):
        splits.append((train_idx, test_idx))
        
        # Log la taille de chaque pli
        logger.debug(f"Pli - Entraînement: {len(train_idx)}, Test: {len(test_idx)}")
    
    logger.info(f"Validation croisée temporelle: {n_splits} plis créés")
    return splits

if __name__ == "__main__":
    # Exemple d'utilisation
    import numpy as np
    
    # Créer des données fictives (série temporelle)
    dates = pd.date_range(start='2020-01-01', periods=1000, freq='H')
    data = pd.DataFrame({
        'timestamp': dates,
        'value': np.sin(np.linspace(0, 20*np.pi, 1000)) + np.random.normal(0, 0.1, 1000),
        'category': np.random.choice(['A', 'B', 'C'], size=1000, p=[0.6, 0.3, 0.1])
    })
    
    # Configuration du logging
    logging.basicConfig(level=logging.INFO, 
                        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    # Exemple 1: Division temporelle classique
    split_result = split_data(data, sequence_length=24, prediction_horizon=12, 
                             method='time', test_size=0.2, validation_size=0.15)
    
    print("Division temporelle:")
    for name, df in split_result.items():
        print(f"  {name}: {len(df)} échantillons")
    
    # Exemple 2: Division stratifiée
    split_result_strat = split_data(data, method='stratified', target_column='category')
    
    print("\nDivision stratifiée:")
    for name, df in split_result_strat.items():
        counts = df['category'].value_counts(normalize=True)
        print(f"  {name}: {len(df)} échantillons, distribution: {dict(counts.round(2))}")
