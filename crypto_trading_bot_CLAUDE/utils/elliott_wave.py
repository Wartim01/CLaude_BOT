"""
Module d'analyse des vagues d'Elliott pour identifier les structures de marché
et les opportunités de trading basées sur la théorie des vagues
"""
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Union
import talib
from scipy import signal

from utils.logger import setup_logger
from utils.technical_analysis import TechnicalAnalysis

logger = setup_logger("elliott_wave")

class ElliottWaveAnalysis:
    """
    Classe pour détecter et analyser les structures de vagues d'Elliott
    dans les données de prix
    """
    def __init__(self):
        """Initialise l'analyseur de vagues d'Elliott"""
        # Configuration
        self.min_wave_size = 0.01  # Taille minimale d'une vague (% du prix)
        self.max_wave_overlap = 0.5  # Chevauchement maximum entre vagues
        self.fibonacci_levels = [0.382, 0.5, 0.618, 0.786, 1.0, 1.618, 2.618]
        
    def detect_waves(self, df: pd.DataFrame, max_degree: int = 3) -> Dict:
        """
        Détecte et analyse les structures de vagues d'Elliott
        
        Args:
            df: DataFrame avec les données OHLCV
            max_degree: Degré maximum des vagues à détecter (1-5)
            
        Returns:
            Dictionnaire avec les structures de vagues détectées
        """
        data = df.copy()
        result = {
            "waves": {},
            "current_structure": "",
            "potential_targets": [],
            "confidence": 0.0
        }
        
        # Vérifier que nous avons suffisamment de données
        if len(data) < 100:
            result["error"] = "Insufficient data for Elliott Wave analysis"
            return result
        
        try:
            # 1. Identifier les pivots importants
            pivots = self._identify_pivots(data)
            
            # 2. Regrouper les pivots en vagues potentielles
            waves = self._group_waves(data, pivots)
            
            # 3. Valider les structures de vagues selon les règles d'Elliott
            impulse, correction = self._validate_wave_structures(data, waves)
            
            # 4. Déterminer la structure actuelle la plus probable
            current_structure, confidence = self._determine_current_structure(data, impulse, correction)
            
            # 5. Calculer les cibles de prix potentielles
            targets = self._calculate_targets(data, current_structure)
            
            # Préparer le résultat
            result["waves"] = {
                "impulse": impulse,
                "correction": correction
            }
            result["current_structure"] = current_structure
            result["confidence"] = confidence
            result["potential_targets"] = targets
            
        except Exception as e:
            result["error"] = f"Error in Elliott Wave analysis: {str(e)}"
        
        return result
    
    def _identify_pivots(self, df: pd.DataFrame) -> Dict:
        """
        Identifie les points pivots dans les données de prix
        
        Args:
            df: DataFrame avec les données OHLCV
            
        Returns:
            Dictionnaire avec les points pivots identifiés
        """
        # Utiliser des fenêtres de différentes tailles pour capturer les pivots de différentes amplitudes
        windows = [5, 10, 20]
        pivots = {"highs": [], "lows": []}
        
        for window in windows:
            # Identifier les pivots avec scipy.signal
            high_idx = signal.argrelextrema(df['high'].values, np.greater, order=window)[0]
            low_idx = signal.argrelextrema(df['low'].values, np.less, order=window)[0]
            
            for idx in high_idx:
                pivots["highs"].append({
                    "index": int(idx),
                    "price": float(df['high'].iloc[idx]),
                    "date": str(df.index[idx]),
                    "window": window
                })
            
            for idx in low_idx:
                pivots["lows"].append({
                    "index": int(idx),
                    "price": float(df['low'].iloc[idx]),
                    "date": str(df.index[idx]),
                    "window": window
                })
        
        # Trier les pivots par index
        pivots["highs"].sort(key=lambda x: x["index"])
        pivots["lows"].sort(key=lambda x: x["index"])
        
        # Éliminer les doublons (pivots très proches)
        pivots["highs"] = self._remove_duplicate_pivots(pivots["highs"])
        pivots["lows"] = self._remove_duplicate_pivots(pivots["lows"])
        
        return pivots
    
    def _remove_duplicate_pivots(self, pivot_list: List[Dict]) -> List[Dict]:
        """
        Élimine les pivots dupliqués ou très proches
        
        Args:
            pivot_list: Liste de pivots
            
        Returns:
            Liste filtrée sans doublons
        """
        if not pivot_list:
            return []
        
        filtered = [pivot_list[0]]
        
        for i in range(1, len(pivot_list)):
            current = pivot_list[i]
            previous = filtered[-1]
            
            # Si les indices sont trop proches, ne conserver que celui avec la fenêtre la plus grande
            if current["index"] - previous["index"] <= 3:
                if current["window"] > previous["window"]:
                    filtered[-1] = current
            else:
                filtered.append(current)
        
        return filtered
    
    def _group_waves(self, df: pd.DataFrame, pivots: Dict) -> Dict:
        """
        Regroupe les pivots en structures de vagues potentielles
        
        Args:
            df: DataFrame avec les données OHLCV
            pivots: Dictionnaire avec les pivots identifiés
            
        Returns:
            Dictionnaire avec les vagues potentielles
        """
        waves = {
            "impulse": [],
            "correction": []
        }
        
        # Créer une liste combinée de tous les pivots
        all_pivots = []
        for p in pivots["highs"]:
            all_pivots.append({"type": "high", **p})
        for p in pivots["lows"]:
            all_pivots.append({"type": "low", **p})
        
        # Trier tous les pivots par index
        all_pivots.sort(key=lambda x: x["index"])
        
        # Chercher des structures de vague d'impulsion (5 vagues)
        for i in range(len(all_pivots) - 5):
            # Une vague d'impulsion devrait suivre la séquence: bas-haut-bas-haut-bas
            if (all_pivots[i]["type"] == "low" and
                all_pivots[i+1]["type"] == "high" and
                all_pivots[i+2]["type"] == "low" and
                all_pivots[i+3]["type"] == "high" and
                all_pivots[i+4]["type"] == "low"):
                
                wave_structure = {
                    "type": "impulse",
                    "start_idx": all_pivots[i]["index"],
                    "end_idx": all_pivots[i+4]["index"],
                    "wave1": {"start": all_pivots[i], "end": all_pivots[i+1]},
                    "wave2": {"start": all_pivots[i+1], "end": all_pivots[i+2]},
                    "wave3": {"start": all_pivots[i+2], "end": all_pivots[i+3]},
                    "wave4": {"start": all_pivots[i+3], "end": all_pivots[i+4]},
                    "wave5": {"start": all_pivots[i+4]},  # Le point de fin de la vague 5 sera déterminé ultérieurement
                    "confidence": 0.0  # Sera calculé lors de la validation
                }
                
                waves["impulse"].append(wave_structure)
        
        # Chercher des structures de vague corrective (3 vagues)
        for i in range(len(all_pivots) - 3):
            # Une vague corrective devrait suivre la séquence: haut-bas-haut
            if (all_pivots[i]["type"] == "high" and
                all_pivots[i+1]["type"] == "low" and
                all_pivots[i+2]["type"] == "high"):
                
                wave_structure = {
                    "type": "correction_abc",
                    "start_idx": all_pivots[i]["index"],
                    "end_idx": all_pivots[i+2]["index"],
                    "waveA": {"start": all_pivots[i], "end": all_pivots[i+1]},
                    "waveB": {"start": all_pivots[i+1], "end": all_pivots[i+2]},
                    "waveC": {"start": all_pivots[i+2]},  # Le point de fin sera déterminé ultérieurement
                    "confidence": 0.0  # Sera calculé lors de la validation
                }
                
                waves["correction"].append(wave_structure)
        
        return waves
    
    def _validate_wave_structures(self, df: pd.DataFrame, waves: Dict) -> Tuple[List[Dict], List[Dict]]:
        """
        Valide les structures de vagues selon les règles d'Elliott
        
        Args:
            df: DataFrame avec les données OHLCV
            waves: Dictionnaire avec les vagues potentielles
            
        Returns:
            Tuple de listes contenant les structures de vagues impulsives et correctives validées
        """
        validated_impulse = []
        validated_correction = []
        
        # Valider les vagues d'impulsion
        for wave in waves["impulse"]:
            score = 0.0
            total_checks = 7.0  # Nombre total de règles à vérifier
            
            # 1. La vague 1 doit être impulsive (haussière)
            wave1_size = wave["wave1"]["end"]["price"] - wave["wave1"]["start"]["price"]
            if wave1_size > 0:
                score += 1.0
            
            # 2. La vague 2 ne doit pas dépasser le début de la vague 1
            if wave["wave2"]["end"]["price"] > wave["wave1"]["start"]["price"]:
                score += 1.0
            
            # 3. La vague 3 doit être plus longue que la vague 1 (règle d'Elliott)
            wave3_size = abs(wave["wave3"]["end"]["price"] - wave["wave3"]["start"]["price"])
            if wave3_size > abs(wave1_size):
                score += 1.0
            
            # 4. La vague 4 ne doit pas entrer dans le territoire de la vague 1
            if wave["wave4"]["end"]["price"] > wave["wave1"]["end"]["price"]:
                score += 1.0
            
            # 5. La vague 5 doit être impulsive (dans la même direction que la vague 1)
            wave5_direction = 1 if wave1_size > 0 else -1
            # Prendre le dernier prix comme fin de la vague 5 si non spécifié
            wave5_end = {"price": df['close'].iloc[-1]} if "end" not in wave["wave5"] else wave["wave5"]["end"]
            wave5_size = wave5_end["price"] - wave["wave5"]["start"]["price"]
            
            if (wave5_size * wave5_direction) > 0:
                score += 1.0
            
            # 6. Vérifier les ratios de Fibonacci entre les vagues
            wave1_price_range = abs(wave1_size)
            wave3_price_range = abs(wave3_size)
            wave5_price_range = abs(wave5_size)
            
            # La vague 3 est souvent à 1.618 ou 2.618 fois la vague 1
            ratio_3_to_1 = wave3_price_range / wave1_price_range if wave1_price_range > 0 else 0
            if abs(ratio_3_to_1 - 1.618) < 0.3 or abs(ratio_3_to_1 - 2.618) < 0.5:
                score += 0.5
            
            # La vague 5 est souvent égale à la vague 1 ou dans un ratio de Fibonacci
            ratio_5_to_1 = wave5_price_range / wave1_price_range if wave1_price_range > 0 else 0
            if abs(ratio_5_to_1 - 1.0) < 0.2 or abs(ratio_5_to_1 - 0.618) < 0.2:
                score += 0.5
            
            # Calculer le score final
            confidence = score / total_checks
            
            # Si le score est suffisamment élevé, ajouter aux vagues validées
            if confidence > 0.6:  # Seuil arbitraire, à ajuster selon les besoins
                wave["confidence"] = confidence
                validated_impulse.append(wave)
        
        # Valider les vagues correctives
        for wave in waves["correction"]:
            score = 0.0
            total_checks = 5.0  # Nombre total de règles à vérifier
            
            # 1. La vague A doit être impulsive (baissière)
            waveA_size = wave["waveA"]["end"]["price"] - wave["waveA"]["start"]["price"]
            if waveA_size < 0:
                score += 1.0
            
            # 2. La vague B doit être corrective et ne pas dépasser le début de A
            waveB_size = wave["waveB"]["end"]["price"] - wave["waveB"]["start"]["price"]
            if waveB_size > 0 and wave["waveB"]["end"]["price"] < wave["waveA"]["start"]["price"]:
                score += 1.0
            
            # 3. La vague C doit être impulsive (baissière)
            waveC_end = {"price": df['close'].iloc[-1]} if "end" not in wave["waveC"] else wave["waveC"]["end"]
            waveC_size = waveC_end["price"] - wave["waveC"]["start"]["price"]
            
            if waveC_size < 0:
                score += 1.0
            
            # 4. Vérifier les ratios de Fibonacci entre les vagues
            waveA_price_range = abs(waveA_size)
            waveC_price_range = abs(waveC_size)
            
            # La vague C est souvent à 1.618 ou 0.618 fois la vague A
            ratio_C_to_A = waveC_price_range / waveA_price_range if waveA_price_range > 0 else 0
            if abs(ratio_C_to_A - 1.618) < 0.3 or abs(ratio_C_to_A - 0.618) < 0.3 or abs(ratio_C_to_A - 1.0) < 0.2:
                score += 1.0
            
            # 5. La vague B retracement est généralement de 50% à 78.6% de la vague A
            if waveA_size != 0:
                retracement = waveB_size / abs(waveA_size)
                if 0.382 <= retracement <= 0.786:
                    score += 1.0
            
            # Calculer le score final
            confidence = score / total_checks
            
            # Si le score est suffisamment élevé, ajouter aux vagues validées
            if confidence > 0.6:  # Seuil arbitraire, à ajuster selon les besoins
                wave["confidence"] = confidence
                validated_correction.append(wave)
        
        return validated_impulse, validated_correction
    
    def _determine_current_structure(self, df: pd.DataFrame, 
                                  impulse_waves: List[Dict], 
                                  correction_waves: List[Dict]) -> Tuple[str, float]:
        """
        Détermine la structure de vague actuelle la plus probable
        
        Args:
            df: DataFrame avec les données OHLCV
            impulse_waves: Liste des vagues impulsives validées
            correction_waves: Liste des vagues correctives validées
            
        Returns:
            Tuple contenant la structure actuelle et le niveau de confiance
        """
        if not impulse_waves and not correction_waves:
            return "unknown", 0.0
        
        current_price = df['close'].iloc[-1]
        latest_idx = df.index[-1]
        
        # Trouver la structure la plus récente
        latest_impulse = None
        for wave in impulse_waves:
            if "end_idx" in wave and wave["end_idx"] > (latest_idx - 20):  # Dans les 20 dernières périodes
                if latest_impulse is None or wave["end_idx"] > latest_impulse["end_idx"]:
                    latest_impulse = wave
        
        latest_correction = None
        for wave in correction_waves:
            if "end_idx" in wave and wave["end_idx"] > (latest_idx - 20):  # Dans les 20 dernières périodes
                if latest_correction is None or wave["end_idx"] > latest_correction["end_idx"]:
                    latest_correction = wave
        
        # Déterminer quelle est la structure la plus récente entre impulsion et correction
        if latest_impulse and latest_correction:
            if latest_impulse["end_idx"] > latest_correction["end_idx"]:
                # L'impulsion est la plus récente, on est probablement dans une correction
                return "completed_impulse", latest_impulse["confidence"]
            else:
                # La correction est la plus récente, on est probablement au début d'une impulsion
                return "completed_correction", latest_correction["confidence"]
        elif latest_impulse:
            # Seulement une impulsion récente détectée
            return "completed_impulse", latest_impulse["confidence"]
        elif latest_correction:
            # Seulement une correction récente détectée
            return "completed_correction", latest_correction["confidence"]
        
        # Si aucune vague récente n'est trouvée, analyser en fonction des tendances récentes
        # Prenons les 20 dernières périodes pour déterminer la tendance
        recent_data = df.iloc[-20:]
        
        # Calculer la direction de la tendance
        price_change = recent_data['close'].iloc[-1] - recent_data['close'].iloc[0]
        
        if price_change > 0:
            # Tendance haussière - pourrait être vague 1, 3, ou 5 d'une impulsion
            # ou vague B d'une correction
            
            # Vérifier si c'est un retracement (pourrait être une vague B)
            prev_trend = df['close'].iloc[-40:-20].mean() - df['close'].iloc[-60:-40].mean()
            
            if prev_trend < 0 and 0.5 < -price_change / prev_trend < 0.786:
                return "wave_b_of_correction", 0.65
            else:
                # Sinon, pourrait être une vague d'impulsion
                return "impulse_wave_in_progress", 0.7
                
        else:  # price_change <= 0
            # Tendance baissière - pourrait être vague 2, 4 d'une impulsion
            # ou vague A, C d'une correction
            
            # Vérifier si c'est un retracement (pourrait être une vague 2 ou 4)
            prev_trend = df['close'].iloc[-40:-20].mean() - df['close'].iloc[-60:-40].mean()
            
            if prev_trend > 0 and 0.5 < -price_change / prev_trend < 0.786:
                return "corrective_wave_in_impulse", 0.65
            else:
                # Sinon, pourrait être une vague de correction (A ou C)
                return "correction_wave_in_progress", 0.7
        
        return "unknown", 0.5
    
    def _calculate_targets(self, df: pd.DataFrame, current_structure: str) -> List[Dict]:
        """
        Calcule les cibles potentielles en fonction de la structure de vagues actuelle
        
        Args:
            df: DataFrame avec les données OHLCV
            current_structure: Structure de vagues actuelle
            
        Returns:
            Liste des cibles de prix potentielles avec leurs niveaux de confiance
        """
        targets = []
        current_price = df['close'].iloc[-1]
        
        # Taille de l'échantillon d'analyse
        lookback = min(200, len(df))
        recent_data = df.iloc[-lookback:]
        
        # Points extrêmes récents
        recent_high = recent_data['high'].max()
        recent_low = recent_data['low'].min()
        price_range = recent_high - recent_low
        
        # En fonction de la structure actuelle, calculer les cibles
        if current_structure == "completed_impulse":
            # Après une impulsion, nous attendons une correction
            # La correction cible généralement des retracements de Fibonacci de la vague impulsive
            
            for fib_level in [0.382, 0.5, 0.618, 0.786]:
                target_price = current_price - (price_range * fib_level)
                confidence = 0.8 if fib_level in [0.5, 0.618] else 0.6
                
                targets.append({
                    "price": float(target_price),
                    "type": f"correction_target_{fib_level}",
                    "confidence": float(confidence),
                    "description": f"Retracement de {fib_level*100}% de la vague impulsive précédente"
                })
        
        elif current_structure == "completed_correction":
            # Après une correction, nous attendons une impulsion dans la direction principale
            # Les extensions de Fibonacci sont utilisées pour projeter les objectifs
            
            for fib_level in [1.0, 1.618, 2.618]:
                target_price = current_price + (price_range * fib_level)
                confidence = 0.8 if fib_level == 1.618 else 0.6
                
                targets.append({
                    "price": float(target_price),
                    "type": f"impulse_target_{fib_level}",
                    "confidence": float(confidence),
                    "description": f"Extension de {fib_level} de la vague impulsive précédente"
                })
        
        elif "impulse_wave_in_progress" in current_structure:
            # Si nous sommes dans une impulsion, la cible dépend de quelle vague nous sommes
            # La vague 3 est souvent la plus forte et la plus étendue
            
            for fib_level in [1.618, 2.618, 3.618]:
                target_price = current_price + (price_range * fib_level / 2)
                confidence = 0.7 if fib_level == 1.618 else 0.5
                
                targets.append({
                    "price": float(target_price),
                    "type": f"impulse_extension_{fib_level}",
                    "confidence": float(confidence),
                    "description": f"Extension de {fib_level} basée sur la structure d'impulsion en cours"
                })
        
        elif "correction_wave_in_progress" in current_structure or "wave_b_of_correction" in current_structure:
            # Dans une correction, la vague C est souvent égale à la vague A ou dans un ratio de Fibonacci
            
            # Calculer approximativement la taille de la vague A
            wave_a_size = (recent_high - recent_low) / 2
            
            for fib_level in [1.0, 1.618]:
                if "wave_b_of_correction" in current_structure:
                    # Si nous sommes dans la vague B, nous ciblons un mouvement à la baisse pour la vague C
                    target_price = current_price - (wave_a_size * fib_level)
                else:
                    # Si nous sommes dans la vague A, nous ciblons un rebond pour la vague B
                    retracement_level = 0.5  # Typique retracement de la vague B
                    target_price = current_price + (wave_a_size * retracement_level)
                
                confidence = 0.65
                targets.append({
                    "price": float(target_price),
                    "type": f"correction_wave_target_{fib_level}",
                    "confidence": float(confidence),
                    "description": f"Projection de vague corrective basée sur la structure actuelle"
                })
        
        # Ajouter également des niveaux de support/résistance clés
        sr_levels = TechnicalAnalysis.detect_support_resistance(df)
        
        if sr_levels.get("nearest_support") is not None and sr_levels["nearest_support"] < current_price:
            targets.append({
                "price": float(sr_levels["nearest_support"]),
                "type": "support",
                "confidence": 0.75,
                "description": "Niveau de support clé"
            })
        
        if sr_levels.get("nearest_resistance") is not None and sr_levels["nearest_resistance"] > current_price:
            targets.append({
                "price": float(sr_levels["nearest_resistance"]),
                "type": "resistance",
                "confidence": 0.75,
                "description": "Niveau de résistance clé"
            })
        
        # Trier les cibles par prix (de bas en haut)
        targets.sort(key=lambda x: x["price"])
        
        return targets
    
    def analyze_wave_count(self, df: pd.DataFrame) -> Dict:
        """
        Fournit une analyse complète du comptage des vagues d'Elliott
        
        Args:
            df: DataFrame avec les données OHLCV
            
        Returns:
            Dictionnaire avec l'analyse des vagues d'Elliott
        """
        # Détecter les vagues
        wave_detection = self.detect_waves(df)
        
        if "error" in wave_detection:
            return wave_detection
        
        # Ajouter des informations supplémentaires
        current_price = df['close'].iloc[-1]
        current_structure = wave_detection["current_structure"]
        confidence = wave_detection["confidence"]
        
        # Déterminer le biais directionnel
        bias = "neutral"
        if current_structure in ["completed_correction", "impulse_wave_in_progress"]:
            bias = "bullish"
        elif current_structure in ["completed_impulse", "correction_wave_in_progress"]:
            bias = "bearish"
        
        # Identifier les niveaux clés à surveiller
        key_levels = []
        for target in wave_detection["potential_targets"]:
            key_levels.append({
                "price": target["price"],
                "type": target["type"],
                "distance_pct": abs(target["price"] - current_price) / current_price * 100
            })
        
        # Trier les niveaux par proximité
        key_levels.sort(key=lambda x: x["distance_pct"])
        
        # Déterminer les recommandations de trading
        trade_recommendations = []
        
        if bias == "bullish" and confidence > 0.6:
            trade_recommendations.append({
                "action": "BUY",
                "target": next((t for t in wave_detection["potential_targets"] 
                                if t["price"] > current_price and "impulse" in t["type"]), 
                               {"price": current_price * 1.1}),
                "stop_loss": next((level["price"] for level in key_levels 
                                  if level["price"] < current_price), current_price * 0.95),
                "reason": f"Structure de vagues {current_structure} avec biais haussier"
            })
        elif bias == "bearish" and confidence > 0.6:
            trade_recommendations.append({
                "action": "SELL",
                "target": next((t for t in wave_detection["potential_targets"] 
                                if t["price"] < current_price and "correction" in t["type"]), 
                               {"price": current_price * 0.9}),
                "stop_loss": next((level["price"] for level in key_levels 
                                  if level["price"] > current_price), current_price * 1.05),
                "reason": f"Structure de vagues {current_structure} avec biais baissier"
            })
        
        # Compléter l'analyse
        analysis = {
            "wave_structures": wave_detection["waves"],
            "current_structure": current_structure,
            "confidence": confidence,
            "bias": bias,
            "potential_targets": wave_detection["potential_targets"],
            "key_levels": key_levels[:3],  # Les 3 niveaux les plus proches
            "trade_recommendations": trade_recommendations,
            "timestamp": pd.Timestamp.now().isoformat()
        }
        
        return analysis
    
    def plot_waves(self, df: pd.DataFrame, analysis: Dict = None):
        """
        Crée un graphique avec les vagues d'Elliott identifiées
        
        Args:
            df: DataFrame avec les données OHLCV
            analysis: Résultat de l'analyse des vagues (si None, une analyse sera faite)
            
        Returns:
            Objet de figure matplotlib
        """
        try:
            import matplotlib.pyplot as plt
            import matplotlib.dates as mdates
            from matplotlib.patches import Rectangle
            
            if analysis is None:
                analysis = self.analyze_wave_count(df)
            
            if "error" in analysis:
                return None
            
            # Créer la figure
            fig, ax = plt.subplots(figsize=(12, 8))
            
            # Tracer les prix
            ax.plot(df.index, df['close'], label='Prix de clôture')
            
            # Tracer les vagues impulsives
            impulse_waves = analysis["wave_structures"].get("impulse", [])
            for wave in impulse_waves:
                if wave["confidence"] > 0.65:
                    # Extraire les indices et prix
                    w1_start_idx = wave["wave1"]["start"]["index"]
                    w1_end_idx = wave["wave1"]["end"]["index"]
                    w3_end_idx = wave["wave3"]["end"]["index"]
                    w5_start_idx = wave["wave5"]["start"]["index"]
                    
                    w1_start_price = wave["wave1"]["start"]["price"]
                    w1_end_price = wave["wave1"]["end"]["price"]
                    w3_end_price = wave["wave3"]["end"]["price"]
                    w5_start_price = wave["wave5"]["start"]["price"]
                    
                    # Tracer les lignes des vagues
                    ax.plot([df.index[w1_start_idx], df.index[w1_end_idx]], 
                           [w1_start_price, w1_end_price], 'g-', linewidth=2)
                    ax.text(df.index[w1_end_idx], w1_end_price, "1", fontsize=12)
                    
                    ax.plot([df.index[w1_end_idx], df.index[w3_end_idx]], 
                           [w1_end_price, w3_end_price], 'g-', linewidth=2)
                    ax.text(df.index[w3_end_idx], w3_end_price, "3", fontsize=12)
                    
                    ax.plot([df.index[w3_end_idx], df.index[w5_start_idx]], 
                           [w3_end_price, w5_start_price], 'g-', linewidth=2)
                    ax.text(df.index[w5_start_idx], w5_start_price, "5", fontsize=12)
            
            # Tracer les vagues correctives
            correction_waves = analysis["wave_structures"].get("correction", [])
            for wave in correction_waves:
                if wave["confidence"] > 0.65:
                    # Extraire les indices et prix
                    wA_start_idx = wave["waveA"]["start"]["index"]
                    wA_end_idx = wave["waveA"]["end"]["index"]
                    wB_end_idx = wave["waveB"]["end"]["index"]
                    wC_start_idx = wave["waveC"]["start"]["index"]
                    
                    wA_start_price = wave["waveA"]["start"]["price"]
                    wA_end_price = wave["waveA"]["end"]["price"]
                    wB_end_price = wave["waveB"]["end"]["price"]
                    wC_start_price = wave["waveC"]["start"]["price"]
                    
                    # Tracer les lignes des vagues
                    ax.plot([df.index[wA_start_idx], df.index[wA_end_idx]], 
                           [wA_start_price, wA_end_price], 'r-', linewidth=2)
                    ax.text(df.index[wA_end_idx], wA_end_price, "A", fontsize=12)
                    
                    ax.plot([df.index[wA_end_idx], df.index[wB_end_idx]], 
                           [wA_end_price, wB_end_price], 'r-', linewidth=2)
                    ax.text(df.index[wB_end_idx], wB_end_price, "B", fontsize=12)
                    
                    ax.plot([df.index[wB_end_idx], df.index[wC_start_idx]], 
                           [wB_end_price, wC_start_price], 'r-', linewidth=2)
                    ax.text(df.index[wC_start_idx], wC_start_price, "C", fontsize=12)
            
            # Tracer les cibles potentielles
            for target in analysis.get("potential_targets", []):
                ax.axhline(y=target["price"], color='blue', linestyle='--', alpha=0.7)
                ax.text(df.index[-1], target["price"], f"{target['type']} ({target['price']:.2f})", fontsize=10)
            
            # Ajouter l'analyse au graphique
            current_price = df['close'].iloc[-1]
            bias = analysis.get("bias", "neutral")
            structure = analysis.get("current_structure", "unknown")
            confidence = analysis.get("confidence", 0)
            
            title_text = f"Analyse des vagues d'Elliott - Structure: {structure} (conf: {confidence:.1%}), Biais: {bias}"
            ax.set_title(title_text)
            
            # Formater l'axe des dates
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
            ax.xaxis.set_major_locator(mdates.WeekdayLocator(interval=2))
            plt.xticks(rotation=45)
            
            # Ajouter une légende
            ax.legend()
            
            plt.tight_layout()
            return fig
            
        except ImportError:
            logger.warning("Matplotlib non disponible pour l'affichage des vagues d'Elliott")
            return None
        except Exception as e:
            logger.error(f"Erreur lors de la création du graphique des vagues: {str(e)}")
            return None