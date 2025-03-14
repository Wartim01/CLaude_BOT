"""
Module de matrice de corrélation pour analyser les relations entre différents actifs
et aider à la gestion des risques du portefeuille
"""
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional, Union
import os
import json
import pickle
from dataclasses import dataclass
from threading import Lock
import time

from config.config import DATA_DIR
from utils.logger import setup_logger

logger = setup_logger("correlation_matrix")

@dataclass
class CorrelationPair:
    """Stocke les informations de corrélation pour une paire d'actifs"""
    symbol1: str
    symbol2: str
    correlation: float
    volatility1: float
    volatility2: float
    timestamp: datetime
    sample_size: int
    time_window: str  # '1d', '7d', '30d', etc.

class CorrelationMatrix:
    """
    Classe pour calculer, stocker et analyser les matrices de corrélation
    entre différents actifs financiers
    """
    def __init__(self, 
                 cache_duration: int = 3600,  # Durée de cache en secondes (1 heure par défaut)
                 min_data_points: int = 30,   # Nb minimum de points de données pour le calcul
                 storage_path: str = None):   # Chemin de stockage personnalisé
        """
        Initialise la matrice de corrélation
        
        Args:
            cache_duration: Durée de validité du cache en secondes
            min_data_points: Nombre minimum de points de données requis pour le calcul
            storage_path: Chemin personnalisé pour le stockage des matrices
        """
        # Configuration
        self.cache_duration = cache_duration
        self.min_data_points = min_data_points
        self.storage_path = storage_path or os.path.join(DATA_DIR, "correlation_matrices")
        
        # Créer le répertoire de stockage si nécessaire
        os.makedirs(self.storage_path, exist_ok=True)
        
        # État interne
        self.matrices = {}  # {timeframe: {timestamp: pd.DataFrame}}
        self.last_update = {}  # {timeframe: timestamp}
        self.correlation_pairs = {}  # {(symbol1, symbol2, timeframe): CorrelationPair}
        self.all_symbols = set()  # Ensemble de tous les symboles connus
        
        # Verrou pour les opérations thread-safe
        self._lock = Lock()
        
        # Charger les données en cache si elles existent
        self._load_cached_data()
        
        logger.info("Matrice de corrélation initialisée")
    
    def update_matrix(self, price_data: Dict[str, pd.DataFrame], 
                      time_window: str = '7d') -> pd.DataFrame:
        """
        Met à jour la matrice de corrélation avec les dernières données de prix
        
        Args:
            price_data: Dictionnaire {symbol: DataFrame avec colonne 'close'}
            time_window: Fenêtre temporelle pour le calcul ('1d', '7d', '30d', etc.)
            
        Returns:
            DataFrame avec la matrice de corrélation mise à jour
        """
        with self._lock:
            current_time = datetime.now()
            
            # Vérifier si une mise à jour est nécessaire (cache expiré)
            last_update = self.last_update.get(time_window, datetime.min)
            if (current_time - last_update).total_seconds() < self.cache_duration:
                # Le cache est encore valide, retourner la matrice existante
                if time_window in self.matrices and self.matrices[time_window]:
                    logger.debug(f"Utilisation du cache pour la matrice {time_window} "
                               f"(expiration dans {self.cache_duration - (current_time - last_update).total_seconds():.0f}s)")
                    return self.matrices[time_window]
            
            # Filtrer les symbols avec données suffisantes
            valid_symbols = {}
            for symbol, df in price_data.items():
                if len(df) >= self.min_data_points:
                    valid_symbols[symbol] = df
                else:
                    logger.warning(f"Données insuffisantes pour {symbol}: {len(df)}/{self.min_data_points} points")
            
            if len(valid_symbols) < 2:
                logger.warning(f"Pas assez de symboles avec des données suffisantes ({len(valid_symbols)})")
                return pd.DataFrame()
            
            # Mettre à jour l'ensemble des symboles
            self.all_symbols.update(valid_symbols.keys())
            
            # Préparer les séries de rendements pour le calcul des corrélations
            returns_dict = {}
            for symbol, df in valid_symbols.items():
                if 'close' in df.columns:
                    # Calculer les rendements logarithmiques
                    returns_dict[symbol] = np.log(df['close'] / df['close'].shift(1)).dropna()
            
            if not returns_dict:
                logger.error("Aucune donnée de prix valide pour calculer les corrélations")
                return pd.DataFrame()
            
            # Créer un DataFrame combiné de tous les rendements
            returns_df = pd.DataFrame(returns_dict)
            
            # Calculer la matrice de corrélation
            try:
                correlation_matrix = returns_df.corr(method='pearson')
                
                # Stocker la matrice mise à jour
                self.matrices[time_window] = correlation_matrix
                self.last_update[time_window] = current_time
                
                # Calculer et stocker les corrélations par paire avec métadonnées
                for i, symbol1 in enumerate(correlation_matrix.index):
                    volatility1 = returns_dict[symbol1].std() * np.sqrt(252)  # Annualisée
                    
                    for j, symbol2 in enumerate(correlation_matrix.columns):
                        if i >= j:  # Éviter les duplications (la matrice est symétrique)
                            continue
                            
                        correlation = correlation_matrix.loc[symbol1, symbol2]
                        volatility2 = returns_dict[symbol2].std() * np.sqrt(252)  # Annualisée
                        
                        # Créer/mettre à jour l'entrée de corrélation
                        pair_key = (symbol1, symbol2, time_window)
                        self.correlation_pairs[pair_key] = CorrelationPair(
                            symbol1=symbol1,
                            symbol2=symbol2,
                            correlation=correlation,
                            volatility1=volatility1,
                            volatility2=volatility2,
                            timestamp=current_time,
                            sample_size=len(returns_df),
                            time_window=time_window
                        )
                
                # Sauvegarder les données mises à jour
                self._save_cached_data()
                
                logger.info(f"Matrice de corrélation mise à jour ({time_window}), {len(correlation_matrix)} actifs")
                return correlation_matrix
                
            except Exception as e:
                logger.error(f"Erreur lors du calcul de la matrice de corrélation: {str(e)}")
                return pd.DataFrame()
    
    def get_correlation(self, symbol1: str, symbol2: str, time_window: str = '7d') -> Optional[float]:
        """
        Récupère la corrélation entre deux actifs
        
        Args:
            symbol1: Premier symbole
            symbol2: Deuxième symbole
            time_window: Fenêtre temporelle
            
        Returns:
            Coefficient de corrélation ou None si non disponible
        """
        # Standardiser l'ordre des symboles pour la recherche
        symbol1, symbol2 = sorted([symbol1, symbol2])
        pair_key = (symbol1, symbol2, time_window)
        
        if pair_key in self.correlation_pairs:
            return self.correlation_pairs[pair_key].correlation
        
        # Vérifier si la matrice existe
        if time_window in self.matrices and self.matrices[time_window] is not None:
            matrix = self.matrices[time_window]
            if symbol1 in matrix.index and symbol2 in matrix.columns:
                return matrix.loc[symbol1, symbol2]
        
        return None
    
    def get_highly_correlated_pairs(self, threshold: float = 0.7, 
                                   time_window: str = '7d') -> List[Tuple[str, str, float]]:
        """
        Retourne les paires d'actifs fortement corrélés
        
        Args:
            threshold: Seuil de corrélation
            time_window: Fenêtre temporelle
            
        Returns:
            Liste de tuples (symbol1, symbol2, correlation)
        """
        result = []
        
        if time_window not in self.matrices or self.matrices[time_window] is None:
            return result
        
        matrix = self.matrices[time_window]
        
        # Parcourir la partie triangulaire supérieure
        for i, symbol1 in enumerate(matrix.index):
            for j, symbol2 in enumerate(matrix.columns):
                if i >= j:  # Éviter les duplications et la diagonale
                    continue
                
                correlation = matrix.loc[symbol1, symbol2]
                if abs(correlation) >= threshold:
                    result.append((symbol1, symbol2, correlation))
        
        # Trier par corrélation descendante
        return sorted(result, key=lambda x: abs(x[2]), reverse=True)
    
    def calculate_portfolio_risk(self, 
                              positions: Dict[str, float], 
                              time_window: str = '7d') -> Dict:
        """
        Calcule les métriques de risque du portefeuille basées sur les corrélations
        
        Args:
            positions: Dictionnaire {symbole: poids} représentant le portefeuille
            time_window: Fenêtre temporelle pour les corrélations
            
        Returns:
            Dict avec métriques de risque et recommandations
        """
        if time_window not in self.matrices or self.matrices[time_window] is None:
            return {
                "error": "Matrice de corrélation non disponible",
                "diversification_score": 0,
                "risk_level": "unknown"
            }
        
        matrix = self.matrices[time_window]
        
        # Filtrer les positions qui sont dans la matrice
        valid_positions = {s: w for s, w in positions.items() if s in matrix.index}
        
        if not valid_positions:
            return {
                "error": "Aucun actif du portefeuille dans la matrice de corrélation",
                "diversification_score": 0,
                "risk_level": "unknown"
            }
        
        # Normaliser les poids
        total_weight = sum(valid_positions.values())
        normalized_weights = {s: w / total_weight for s, w in valid_positions.items()} if total_weight > 0 else valid_positions
        
        # Créer le vecteur de poids et extraire la sous-matrice
        symbols = list(valid_positions.keys())
        weights = np.array([normalized_weights[s] for s in symbols])
        sub_matrix = matrix.loc[symbols, symbols]
        
        # Calculer la diversification du portefeuille
        weighted_correlation = 0.0
        high_correlation_exposure = 0.0
        
        for i, si in enumerate(symbols):
            for j, sj in enumerate(symbols):
                if i != j:
                    pair_corr = sub_matrix.loc[si, sj]
                    weight = weights[i] * weights[j]
                    weighted_correlation += pair_corr * weight
                    
                    if pair_corr > 0.7:  # Haute corrélation
                        high_correlation_exposure += weight
        
        # Normaliser score de diversification (1 = parfaite, 0 = aucune)
        diversification_score = max(0, 1 - (weighted_correlation / 2))
        
        # Identifie les paires à haute corrélation dans le portefeuille
        high_corr_pairs = []
        for i, si in enumerate(symbols):
            for j, sj in enumerate(symbols):
                if i < j:  # Éviter les duplications
                    corr = sub_matrix.loc[si, sj]
                    if abs(corr) > 0.7:
                        high_corr_pairs.append((si, sj, corr, normalized_weights[si], normalized_weights[sj]))
        
        # Trier par exposition (poids combinés) décroissante
        high_corr_pairs.sort(key=lambda x: (x[3] + x[4]) * abs(x[2]), reverse=True)
        
        # Déterminer le niveau de risque
        if diversification_score > 0.8:
            risk_level = "low"
        elif diversification_score > 0.6:
            risk_level = "moderate"
        elif diversification_score > 0.4:
            risk_level = "elevated"
        else:
            risk_level = "high"
        
        # Générer des recommandations
        recommendations = []
        
        if high_corr_pairs:
            # Recommander la réduction des positions hautement corrélées
            for pair in high_corr_pairs[:3]:  # Top 3 des paires les plus risquées
                si, sj, corr, wi, wj = pair
                # Déterminer lequel réduire (celui avec le poids le plus élevé ou celui avec la volatilité la plus élevée)
                to_reduce = si if wi > wj else sj
                recommendations.append(f"Réduire la position {to_reduce} (corrélation de {corr:.2f} avec {si if to_reduce == sj else sj})")
        
        if diversification_score < 0.5:
            recommendations.append("Améliorer la diversification en ajoutant des actifs peu corrélés")
        
        return {
            "diversification_score": diversification_score,
            "risk_level": risk_level,
            "high_correlation_exposure": high_correlation_exposure,
            "high_correlation_pairs": [(p[0], p[1], p[2]) for p in high_corr_pairs],
            "recommendations": recommendations
        }
    
    def visualize_matrix(self, time_window: str = '7d', 
                       save_path: Optional[str] = None,
                       show_plot: bool = True) -> None:
        """
        Crée une visualisation de la matrice de corrélation
        
        Args:
            time_window: Fenêtre temporelle
            save_path: Chemin pour sauvegarder l'image
            show_plot: Afficher le graphique
        """
        if time_window not in self.matrices or self.matrices[time_window] is None:
            logger.error(f"Matrice de corrélation non disponible pour {time_window}")
            return
        
        matrix = self.matrices[time_window]
        if matrix.empty:
            logger.error(f"Matrice de corrélation vide pour {time_window}")
            return
        
        try:
            # Définir la taille du graphique en fonction du nombre d'actifs
            n_assets = len(matrix)
            figsize = (max(8, n_assets * 0.4), max(6, n_assets * 0.4))
            
            plt.figure(figsize=figsize)
            mask = np.triu(np.ones_like(matrix, dtype=bool))  # Masquer la moitié supérieure
            
            # Créer la heatmap avec Seaborn
            cmap = sns.diverging_palette(230, 20, as_cmap=True)
            ax = sns.heatmap(
                matrix, 
                mask=mask,
                cmap=cmap,
                vmax=1.0,
                vmin=-1.0,
                center=0,
                square=True,
                linewidths=0.5,
                cbar_kws={"shrink": 0.8, "label": "Coefficient de Corrélation"},
                annot=True if n_assets < 15 else False,  # Afficher les valeurs si peu d'actifs
                fmt=".2f" if n_assets < 15 else None
            )
            
            # Paramétrer le graphique
            plt.title(f"Matrice de Corrélation des Actifs ({time_window})")
            plt.tight_layout()
            
            # Sauvegarder si demandé
            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches="tight")
                logger.info(f"Visualisation sauvegardée: {save_path}")
            
            # Afficher si demandé
            if show_plot:
                plt.show()
            else:
                plt.close()
                
        except Exception as e:
            logger.error(f"Erreur lors de la visualisation de la matrice: {str(e)}")
            plt.close()
    
    def generate_correlation_report(self, time_window: str = '7d',
                                  high_corr_threshold: float = 0.7,
                                  low_corr_threshold: float = 0.3,
                                  save_path: Optional[str] = None) -> Dict:
        """
        Génère un rapport détaillé sur les corrélations
        
        Args:
            time_window: Fenêtre temporelle
            high_corr_threshold: Seuil pour les corrélations élevées
            low_corr_threshold: Seuil pour les corrélations faibles
            save_path: Chemin pour sauvegarder le rapport en JSON
            
        Returns:
            Dictionnaire avec le rapport de corrélation
        """
        if time_window not in self.matrices or self.matrices[time_window] is None:
            return {"error": f"Matrice de corrélation non disponible pour {time_window}"}
        
        matrix = self.matrices[time_window]
        
        # Préparer le rapport
        report = {
            "timestamp": datetime.now().isoformat(),
            "time_window": time_window,
            "asset_count": len(matrix),
            "update_time": self.last_update.get(time_window, datetime.min).isoformat(),
            "high_correlations": [],
            "low_correlations": [],
            "negative_correlations": [],
            "average_correlation": float(matrix.values[np.triu_indices_from(matrix.values, k=1)].mean()),
            "diversity_potential": {}
        }
        
        # Trouver les fortes corrélations
        for i, symbol1 in enumerate(matrix.index):
            for j, symbol2 in enumerate(matrix.columns):
                if i >= j:  # Éviter les duplications
                    continue
                
                corr = matrix.loc[symbol1, symbol2]
                
                # Collecter les paires selon leur niveau de corrélation
                if corr >= high_corr_threshold:
                    report["high_correlations"].append({
                        "pair": [symbol1, symbol2],
                        "correlation": float(corr)
                    })
                elif abs(corr) <= low_corr_threshold:
                    report["low_correlations"].append({
                        "pair": [symbol1, symbol2],
                        "correlation": float(corr)
                    })
                elif corr <= -high_corr_threshold:  # Corrélations fortement négatives
                    report["negative_correlations"].append({
                        "pair": [symbol1, symbol2],
                        "correlation": float(corr)
                    })
        
        # Trier les listes par corrélation (abs pour les corrélations négatives)
        report["high_correlations"].sort(key=lambda x: x["correlation"], reverse=True)
        report["low_correlations"].sort(key=lambda x: abs(x["correlation"]))
        report["negative_correlations"].sort(key=lambda x: x["correlation"])
        
        # Calculer le potentiel de diversification
        for symbol in matrix.index:
            # Pour chaque symbole, trouver les actifs les moins corrélés
            correlations = [(other, matrix.loc[symbol, other]) 
                           for other in matrix.columns if other != symbol]
            
            # Trier par corrélation croissante
            correlations.sort(key=lambda x: x[1])
            
            # Garder les 5 meilleurs diversificateurs
            best_diversifiers = correlations[:5]
            
            report["diversity_potential"][symbol] = [
                {"symbol": div[0], "correlation": float(div[1])}
                for div in best_diversifiers
            ]
        
        # Sauvegarder le rapport si demandé
        if save_path:
            try:
                with open(save_path, 'w') as f:
                    json.dump(report, f, indent=2)
                logger.info(f"Rapport de corrélation sauvegardé: {save_path}")
            except Exception as e:
                logger.error(f"Erreur lors de la sauvegarde du rapport: {str(e)}")
        
        return report
    
    def _save_cached_data(self) -> None:
        """
        Sauvegarde les données en cache pour une utilisation future
        """
        try:
            # Préparer les données à sauvegarder
            cache_data = {
                "matrices": {k: v.to_dict() if isinstance(v, pd.DataFrame) else None 
                            for k, v in self.matrices.items()},
                "last_update": {k: v.isoformat() for k, v in self.last_update.items()},
                "correlation_pairs": self.correlation_pairs,
                "all_symbols": list(self.all_symbols),
                "timestamp": datetime.now().isoformat()
            }
            
            # Sauvegarder au format pickle
            cache_path = os.path.join(self.storage_path, "correlation_cache.pkl")
            with open(cache_path, 'wb') as f:
                pickle.dump(cache_data, f)
            
            # Sauvegarder aussi le résumé au format JSON
            summary_data = {
                "time_windows": list(self.matrices.keys()),
                "last_update": {k: v.isoformat() for k, v in self.last_update.items()},
                "asset_count": {k: len(v.index) if isinstance(v, pd.DataFrame) else 0 
                               for k, v in self.matrices.items()},
                "all_symbols": list(self.all_symbols),
                "timestamp": datetime.now().isoformat()
            }
            
            summary_path = os.path.join(self.storage_path, "correlation_summary.json")
            with open(summary_path, 'w') as f:
                json.dump(summary_data, f, indent=2)
                
            logger.debug("Données de corrélation mises en cache")
            
        except Exception as e:
            logger.error(f"Erreur lors de la sauvegarde des données en cache: {str(e)}")
    
    def _load_cached_data(self) -> bool:
        """
        Charge les données en cache
        
        Returns:
            Succès du chargement
        """
        cache_path = os.path.join(self.storage_path, "correlation_cache.pkl")
        
        if not os.path.exists(cache_path):
            logger.debug("Pas de données de corrélation en cache")
            return False
            
        try:
            with open(cache_path, 'rb') as f:
                cache_data = pickle.load(f)
            
            # Restaurer les matrices
            self.matrices = {}
            for k, v in cache_data["matrices"].items():
                if v is not None:
                    self.matrices[k] = pd.DataFrame.from_dict(v)
                else:
                    self.matrices[k] = None
            
            # Restaurer les horodatages
            self.last_update = {k: datetime.fromisoformat(v) 
                              for k, v in cache_data["last_update"].items()}
            
            # Restaurer les autres données
            self.correlation_pairs = cache_data["correlation_pairs"]
            self.all_symbols = set(cache_data["all_symbols"])
            
            logger.info(f"Données de corrélation chargées depuis le cache, "
                      f"{len(self.matrices)} matrices, {len(self.all_symbols)} symboles")
            return True
            
        except Exception as e:
            logger.error(f"Erreur lors du chargement des données en cache: {str(e)}")
            return False

    def update(self, market_data: dict) -> None:
        """
        Update the correlation matrix from provided market data.
        Args:
            market_data: Dictionary mapping asset symbol to DataFrame with a 'close' column.
        """
        correlations = {}
        symbols = list(market_data.keys())
        for i in range(len(symbols)):
            for j in range(i+1, len(symbols)):
                sym1 = symbols[i]
                sym2 = symbols[j]
                df1 = market_data[sym1]
                df2 = market_data[sym2]
                # Align series using the most recent common period
                min_length = min(len(df1), len(df2))
                if min_length < 2:
                    corr = np.nan
                else:
                    series1 = df1['close'].iloc[-min_length:]
                    series2 = df2['close'].iloc[-min_length:]
                    corr = np.corrcoef(series1, series2)[0, 1]
                correlations[(sym1, sym2)] = corr
                correlations[(sym2, sym1)] = corr
        self.matrix = correlations
        self.last_update = time.time()

    def get_correlation(self, asset1: str, asset2: str, market_data: dict) -> float:
        """
        Get the correlation between two assets.
        If the cache is expired, recalculates the correlation matrix.
        
        Args:
            asset1: First asset symbol.
            asset2: Second asset symbol.
            market_data: Dictionary of market data (used for update if cache expired).
        
        Returns:
            Correlation value (float) or np.nan if not available.
        """
        current_time = time.time()
        if current_time - self.last_update > self.cache_duration:
            self.update(market_data)
        return self.matrix.get((asset1, asset2), np.nan)
