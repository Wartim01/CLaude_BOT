# utils/model_explainer.py
"""
Module pour expliquer les décisions du modèle LSTM et générer des insights sur les facteurs
qui influencent les prédictions
"""
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional, Union
from datetime import datetime, timedelta
import json
import shap
import pickle
from sklearn.preprocessing import MinMaxScaler

from config.config import DATA_DIR
from ai.models.lstm_model import LSTMModel
from ai.models.feature_engineering import FeatureEngineering
from utils.logger import setup_logger
from ai.reasoning_engine import ReasoningEngine

logger = setup_logger("model_explainer")

class ModelExplainer:
    """
    Classe pour l'explication des décisions du modèle LSTM
    Fournit des insights sur les facteurs qui influencent les prédictions et
    génère des visualisations explicatives
    """
    def __init__(self, model_path: Optional[str] = None, feature_engineering: Optional[FeatureEngineering] = None):
        """
        Initialise l'explainer
        
        Args:
            model_path: Chemin vers le modèle LSTM
            feature_engineering: Module d'ingénierie des caractéristiques
        """
        self.output_dir = os.path.join(DATA_DIR, "model_explanations")
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Charger ou initialiser le modèle LSTM
        self.model = None
        if model_path:
            try:
                self.model = LSTMModel()
                self.model.load(model_path)
                logger.info(f"Modèle LSTM chargé: {model_path}")
            except Exception as e:
                logger.error(f"Erreur lors du chargement du modèle: {str(e)}")
        
        # Initialiser le module d'ingénierie des caractéristiques
        self.feature_engineering = feature_engineering or FeatureEngineering()
        
        # Initialiser le moteur de raisonnement pour les explications textuelles
        self.reasoning_engine = ReasoningEngine()
        
        # Générateur d'explications SHAP
        self.shap_explainer = None
        
        # Cache pour les données normalisées récentes
        self.recent_data_cache = {}
    
    def explain_prediction(self, data: pd.DataFrame, 
                         prediction: Dict, 
                         generate_visualizations: bool = True,
                         output_prefix: Optional[str] = None) -> Dict:
        """
        Explique une prédiction du modèle LSTM
        
        Args:
            data: DataFrame avec les données OHLCV
            prediction: Prédiction du modèle à expliquer
            generate_visualizations: Générer des visualisations explicatives
            output_prefix: Préfixe pour les fichiers de sortie
            
        Returns:
            Dictionnaire avec les explications
        """
        if self.model is None:
            logger.error("Aucun modèle LSTM n'a été chargé")
            return {"success": False, "error": "Modèle non initialisé"}
        
        try:
            # 1. Préparer les données
            featured_data, normalized_data = self._prepare_data(data)
            
            # 2. Identifier les facteurs importants
            important_factors = self._identify_important_factors(normalized_data, prediction)
            
            # 3. Générer une explication textuelle
            textual_explanation = self._generate_textual_explanation(prediction, important_factors, data)
            
            # 4. Générer des visualisations si demandé
            visualizations = {}
            if generate_visualizations:
                visualizations = self._generate_visualizations(
                    data=data,
                    featured_data=featured_data,
                    normalized_data=normalized_data,
                    prediction=prediction,
                    important_factors=important_factors,
                    output_prefix=output_prefix
                )
            
            # 5. Assembler les explications
            explanation = {
                "success": True,
                "timestamp": datetime.now().isoformat(),
                "prediction": prediction,
                "textual_explanation": textual_explanation,
                "important_factors": important_factors,
                "visualizations": visualizations if generate_visualizations else None
            }
            
            # 6. Sauvegarder l'explication si un préfixe est fourni
            if output_prefix:
                output_file = f"{output_prefix}_explanation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
                output_path = os.path.join(self.output_dir, output_file)
                
                with open(output_path, 'w') as f:
                    json.dump(explanation, f, indent=2, default=str)
                logger.info(f"Explication sauvegardée: {output_path}")
            
            return explanation
        
        except Exception as e:
            logger.error(f"Erreur lors de l'explication de la prédiction: {str(e)}")
            import traceback
            traceback.print_exc()
            return {"success": False, "error": str(e)}
    
    def explain_trade_decision(self, data: pd.DataFrame, opportunity: Dict,
                              generate_visualizations: bool = True) -> Dict:
        """
        Explique une décision de trading basée sur le modèle LSTM
        
        Args:
            data: DataFrame avec les données OHLCV
            opportunity: Opportunité de trading à expliquer
            generate_visualizations: Générer des visualisations explicatives
            
        Returns:
            Dictionnaire avec les explications
        """
        # 1. Extraire les prédictions LSTM de l'opportunité
        lstm_prediction = opportunity.get("lstm_prediction")
        
        if not lstm_prediction:
            return {
                "success": False,
                "error": "Aucune prédiction LSTM dans l'opportunité fournie"
            }
        
        # 2. Générer l'explication des prédictions
        prediction_explanation = self.explain_prediction(
            data=data,
            prediction=lstm_prediction,
            generate_visualizations=generate_visualizations,
            output_prefix=f"{opportunity.get('symbol', 'unknown')}_trade"
        )
        
        # 3. Générer une explication spécifique à la décision de trading
        trade_explanation = {
            "success": prediction_explanation["success"],
            "timestamp": datetime.now().isoformat(),
            "opportunity": opportunity,
            "prediction_explanation": prediction_explanation,
            "trade_reasoning": self._explain_trade_opportunity(opportunity)
        }
        
        # 4. Générer des visualisations de la décision de trading
        if generate_visualizations:
            trade_explanation["trade_visualizations"] = self._visualize_trade_opportunity(
                data=data,
                opportunity=opportunity
            )
        
        return trade_explanation
    
    def _prepare_data(self, data: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Prépare les données pour l'explication
        
        Args:
            data: DataFrame avec les données OHLCV
            
        Returns:
            Tuple (featured_data, normalized_data)
        """
        # Vérifier si les données sont dans le cache
        data_key = data.index[-1].isoformat()
        if data_key in self.recent_data_cache:
            return self.recent_data_cache[data_key]
        
        # Créer les caractéristiques avancées
        featured_data = self.feature_engineering.create_features(
            data,
            include_time_features=True,
            include_price_patterns=True
        )
        
        # Normaliser les caractéristiques
        normalized_data = self.feature_engineering.scale_features(
            featured_data,
            is_training=False,
            method='standard',
            feature_group='lstm'
        )
        
        # Stocker dans le cache
        self.recent_data_cache[data_key] = (featured_data, normalized_data)
        
        # Limiter la taille du cache
        if len(self.recent_data_cache) > 10:
            # Supprimer la plus ancienne entrée
            oldest_key = min(self.recent_data_cache.keys())
            del self.recent_data_cache[oldest_key]
        
        return featured_data, normalized_data
    
    def _identify_important_factors(self, normalized_data: pd.DataFrame, 
                                  prediction: Dict) -> Dict:
        """
        Identifie les facteurs importants qui ont influencé la prédiction
        
        Args:
            normalized_data: DataFrame des données normalisées
            prediction: Prédiction du modèle
            
        Returns:
            Dictionnaire des facteurs importants par horizon
        """
        important_factors = {}
        
        # Si SHAP n'est pas encore initialisé
        if self.shap_explainer is None and self.model is not None:
            try:
                # Initialiser SHAP pour le modèle LSTM
                self._initialize_shap(normalized_data)
            except Exception as e:
                logger.warning(f"Impossible d'initialiser SHAP: {str(e)}")
        
        # Pour chaque horizon de prédiction
        for horizon_key, horizon_prediction in prediction.items():
            # Si nous avons un explainer SHAP, l'utiliser
            if self.shap_explainer is not None:
                # Préparer les données pour l'explication SHAP
                X = self._prepare_data_for_shap(normalized_data)
                
                # Calculer les valeurs SHAP
                try:
                    shap_values = self.shap_explainer.shap_values(X)
                    
                    # Identifier les caractéristiques importantes
                    if isinstance(shap_values, list):
                        # Plusieurs sorties, trouver l'indice correspondant à l'horizon actuel
                        horizon_idx = int(horizon_key.split('_')[-1]) if '_' in horizon_key else 0
                        horizon_idx = min(horizon_idx, len(shap_values) - 1)
                        values = shap_values[horizon_idx]
                    else:
                        values = shap_values
                    
                    # Moyenner les valeurs SHAP sur la séquence
                    mean_values = np.mean(np.abs(values), axis=1)
                    
                    # Obtenir les noms des caractéristiques
                    feature_names = normalized_data.columns.tolist()
                    
                    # Trier par importance
                    importance_scores = [(feature_names[i], float(mean_values[0, i])) 
                                        for i in range(min(len(feature_names), mean_values.shape[1]))]
                    importance_scores.sort(key=lambda x: x[1], reverse=True)
                    
                    # Conserver les 10 caractéristiques les plus importantes
                    top_features = importance_scores[:10]
                    
                    important_factors[horizon_key] = {
                        "method": "shap",
                        "top_features": top_features
                    }
                
                except Exception as e:
                    logger.warning(f"Erreur lors du calcul des valeurs SHAP: {str(e)}")
                    # Fallback à une méthode plus simple
                    important_factors[horizon_key] = self._fallback_feature_importance(normalized_data, horizon_prediction)
            
            else:
                # Méthode alternative d'importance des caractéristiques
                important_factors[horizon_key] = self._fallback_feature_importance(normalized_data, horizon_prediction)
        
        return important_factors
    
    def _fallback_feature_importance(self, normalized_data: pd.DataFrame, 
                                   horizon_prediction: Dict) -> Dict:
        """
        Méthode alternative pour estimer l'importance des caractéristiques
        
        Args:
            normalized_data: DataFrame des données normalisées
            horizon_prediction: Prédiction pour un horizon spécifique
            
        Returns:
            Dictionnaire des caractéristiques importantes
        """
        # Cette méthode utilise des heuristiques basées sur la corrélation et les tendances
        
        # Récupérer les colonnes numériques
        numeric_columns = normalized_data.select_dtypes(include=['float64', 'int64']).columns.tolist()
        
        # Caractéristiques connues pour être importantes
        important_indicators = [
            col for col in numeric_columns if any(
                keyword in col.lower() for keyword in 
                ['rsi', 'macd', 'bb_', 'ema', 'adx', 'atr', 'momentum', 'trend']
            )
        ]
        
        # Calculer la corrélation avec le prix de clôture
        correlations = normalized_data[numeric_columns].corrwith(normalized_data['close']).abs()
        
        # Trier par corrélation
        sorted_correlations = correlations.sort_values(ascending=False)
        
        # Fusionner avec les indicateurs importants
        top_features = []
        
        # Ajouter d'abord les indicateurs importants
        for indicator in important_indicators:
            if indicator in sorted_correlations:
                correlation = sorted_correlations[indicator]
                top_features.append((indicator, float(correlation)))
        
        # Ajouter ensuite les autres caractéristiques par ordre de corrélation
        for feature, correlation in sorted_correlations.items():
            if feature not in [f[0] for f in top_features]:
                top_features.append((feature, float(correlation)))
                if len(top_features) >= 10:
                    break
        
        return {
            "method": "correlation_heuristic",
            "top_features": top_features[:10]
        }
    
    def _initialize_shap(self, sample_data: pd.DataFrame) -> None:
        """
        Initialise le explainer SHAP pour le modèle LSTM
        
        Args:
            sample_data: Échantillon de données pour initialiser l'explainer
        """
        try:
            # Préparer un échantillon de données pour SHAP
            X = self._prepare_data_for_shap(sample_data)
            
            # Créer un wrapper pour le modèle qui retourne une sortie unique
            def model_wrapper(x):
                # Convertir les données au format attendu par le modèle
                if len(x.shape) == 3:  # (n_samples, seq_len, n_features)
                    x_tensor = x
                else:  # (n_samples, seq_len * n_features)
                    n_samples = x.shape[0]
                    seq_len = self.model.input_length
                    n_features = sample_data.shape[1]
                    x_tensor = x.reshape(n_samples, seq_len, n_features)
                
                # Faire une prédiction
                predictions = self.model.model.predict(x_tensor)
                
                # Retourner seulement la première sortie (direction) pour simplifier
                return predictions[0]
            
            # Initialiser l'explainer SHAP
            self.shap_explainer = shap.DeepExplainer(model_wrapper, X)
            logger.info("Explainer SHAP initialisé avec succès")
            
        except Exception as e:
            logger.error(f"Erreur lors de l'initialisation de SHAP: {str(e)}")
            self.shap_explainer = None
    
    def _prepare_data_for_shap(self, data: pd.DataFrame) -> np.ndarray:
        """
        Prépare les données au format attendu par SHAP
        
        Args:
            data: DataFrame des données normalisées
            
        Returns:
            Tableau numpy au format attendu par SHAP
        """
        # Créer les séquences pour le modèle LSTM
        X, _ = self.feature_engineering.create_multi_horizon_data(
            data,
            sequence_length=self.model.input_length,
            horizons=self.model.horizon_periods,
            is_training=False
        )
        
        # S'assurer que X a la bonne forme (n_samples, seq_len, n_features)
        if len(X) == 0:
            # Créer un échantillon vide avec la bonne forme
            X = np.zeros((1, self.model.input_length, data.shape[1]))
        elif len(X.shape) == 2:
            # Ajouter la dimension du batch
            X = np.expand_dims(X, axis=0)
        
        return X
    
    def _generate_textual_explanation(self, prediction: Dict, 
                                    important_factors: Dict,
                                    original_data: pd.DataFrame) -> Dict:
        """
        Génère une explication textuelle de la prédiction
        
        Args:
            prediction: Prédiction du modèle
            important_factors: Facteurs importants par horizon
            original_data: Données OHLCV originales
            
        Returns:
            Dictionnaire avec les explications textuelles par horizon
        """
        explanations = {}
        
        # Prix actuel
        current_price = original_data['close'].iloc[-1]
        
        # Pour chaque horizon de prédiction
        for horizon_key, horizon_prediction in prediction.items():
            # Extraire les prédictions clés
            direction_prob = horizon_prediction.get("direction_probability", 50)
            predicted_momentum = horizon_prediction.get("predicted_momentum", 0)
            predicted_volatility = horizon_prediction.get("predicted_volatility", 0)
            
            # Déterminer la direction prédite
            if direction_prob > 60:
                direction = "haussière"
                confidence = "forte" if direction_prob > 80 else "modérée"
            elif direction_prob < 40:
                direction = "baissière"
                confidence = "forte" if direction_prob < 20 else "modérée"
            else:
                direction = "neutre"
                confidence = "faible"
            
            # Déterminer la volatilité attendue
            volatility_level = "élevée" if predicted_volatility > 1.2 else "normale" if predicted_volatility > 0.8 else "faible"
            
            # Déterminer le momentum
            momentum_desc = "fort positif" if predicted_momentum > 0.6 else \
                           "positif" if predicted_momentum > 0.2 else \
                           "fort négatif" if predicted_momentum < -0.6 else \
                           "négatif" if predicted_momentum < -0.2 else \
                           "neutre"
            
            # Construire l'explication de base
            horizon_name = horizon_key.replace("horizon_", "").replace("_", " ").upper() if "_" in horizon_key else horizon_key
            base_explanation = (
                f"Pour l'horizon {horizon_name}, le modèle prédit une tendance {direction} "
                f"avec une confiance {confidence} ({direction_prob:.1f}%). "
                f"Le momentum attendu est {momentum_desc} et la volatilité prévue est {volatility_level}."
            )
            
            # Ajouter des détails sur les facteurs importants
            factors = important_factors.get(horizon_key, {}).get("top_features", [])
            if factors:
                factors_text = "Les facteurs les plus influents dans cette prédiction sont: "
                factors_list = [f"{name} (impact: {score:.3f})" for name, score in factors[:5]]
                factors_text += ", ".join(factors_list)
                
                base_explanation += f" {factors_text}."
            
            # Estimer le changement de prix attendu
            price_change_estimate = self._estimate_price_change(
                direction_prob=direction_prob,
                momentum=predicted_momentum,
                volatility=predicted_volatility,
                horizon=horizon_key
            )
            
            target_price = current_price * (1 + price_change_estimate/100)
            
            price_change_text = (
                f"Selon ces prédictions, le prix pourrait évoluer d'environ {price_change_estimate:.2f}% "
                f"par rapport au prix actuel de {current_price:.2f}, "
                f"soit un objectif potentiel vers {target_price:.2f}."
            )
            
            base_explanation += f" {price_change_text}"
            
            # Ajouter les recommandations d'action
            action_recommendation = self._get_action_recommendation(
                direction_prob=direction_prob,
                momentum=predicted_momentum,
                volatility=predicted_volatility
            )
            
            # Assembler l'explication complète
            explanations[horizon_key] = {
                "summary": base_explanation,
                "direction": direction,
                "confidence": confidence,
                "momentum": momentum_desc,
                "volatility": volatility_level,
                "price_change_estimate": price_change_estimate,
                "target_price": target_price,
                "action_recommendation": action_recommendation
            }
        
        # Ajouter une conclusion globale
        explanations["conclusion"] = self._generate_global_conclusion(prediction, explanations)
        
        return explanations
    
    def _estimate_price_change(self, direction_prob: float, momentum: float, 
                             volatility: float, horizon: str) -> float:
        """
        Estime le changement de prix attendu basé sur les prédictions
        
        Args:
            direction_prob: Probabilité de direction (0-100)
            momentum: Momentum prédit (-1 à 1)
            volatility: Volatilité prédite (généralement 0.5-1.5)
            horizon: Clé de l'horizon de prédiction
            
        Returns:
            Estimation du changement de prix en pourcentage
        """
        # Facteur de base basé sur la probabilité de direction
        # Convertir la probabilité (0-100) en un facteur (-1 à 1)
        direction_factor = (direction_prob - 50) / 50
        
        # Combiner avec le momentum
        combined_signal = (direction_factor + momentum) / 2
        
        # Ajuster par la volatilité
        # Une volatilité plus élevée implique des mouvements plus importants
        price_change = combined_signal * volatility * 5  # 5% est un mouvement type pour volatilité=1
        
        # Ajuster en fonction de l'horizon
        horizon_multiplier = 1.0
        if "12" in horizon or "short" in horizon:
            horizon_multiplier = 1.0
        elif "24" in horizon or "mid" in horizon:
            horizon_multiplier = 1.5
        elif "96" in horizon or "long" in horizon:
            horizon_multiplier = 2.5
        
        return price_change * horizon_multiplier
    
    def _get_action_recommendation(self, direction_prob: float, 
                                 momentum: float, 
                                 volatility: float) -> str:
        """
        Génère une recommandation d'action basée sur les prédictions
        
        Args:
            direction_prob: Probabilité de direction (0-100)
            momentum: Momentum prédit (-1 à 1)
            volatility: Volatilité prédite
            
        Returns:
            Recommandation d'action
        """
        # Signal fort à la hausse
        if direction_prob > 75 and momentum > 0.3:
            if volatility > 1.2:
                return "ACHAT AGRESSIF avec stop-loss adapté à la forte volatilité"
            else:
                return "ACHAT avec stop-loss standard"
        
        # Signal modéré à la hausse
        elif direction_prob > 60 and momentum > 0:
            return "ACHAT PRUDENT avec prise de position partielle"
        
        # Signal fort à la baisse
        elif direction_prob < 25 and momentum < -0.3:
            if volatility > 1.2:
                return "VENTE AGRESSIVE avec stop-loss adapté à la forte volatilité"
            else:
                return "VENTE avec stop-loss standard"
        
        # Signal modéré à la baisse
        elif direction_prob < 40 and momentum < 0:
            return "VENTE PRUDENTE avec prise de position partielle"
        
        # Volatilité élevée sans direction claire
        elif volatility > 1.3 and 40 <= direction_prob <= 60:
            return "ATTENTE - volatilité élevée sans direction claire"
        
        # Signal neutre
        else:
            return "OBSERVATION - signal insuffisant pour une action claire"
    
    def _generate_global_conclusion(self, prediction: Dict, explanations: Dict) -> str:
        """
        Génère une conclusion globale intégrant les prédictions de tous les horizons
        
        Args:
            prediction: Prédiction complète
            explanations: Explications par horizon
            
        Returns:
            Conclusion globale
        """
        # Identifier les horizons court, moyen et long terme
        short_term = None
        mid_term = None
        long_term = None
        
        for key in prediction.keys():
            if "12" in key or "short" in key:
                short_term = key
            elif "24" in key or "mid" in key:
                mid_term = key
            elif "96" in key or "long" in key:
                long_term = key
        
        # Construire la conclusion en fonction des tendances sur différents horizons
        conclusion = "Analyse multi-horizon: "
        
        # Vérifier la cohérence des signaux
        signals_aligned = True
        directions = []
        
        for key in [short_term, mid_term, long_term]:
            if key and key in explanations:
                directions.append(explanations[key]["direction"])
        
        # Des signaux sont alignés si tous pointent dans la même direction
        signals_aligned = len(set(directions)) == 1 and len(directions) > 1
        
        if signals_aligned:
            direction = directions[0]
            conclusion += f"Les signaux sont alignés sur tous les horizons, indiquant une tendance {direction} cohérente. "
            
            if direction == "haussière":
                conclusion += "Cette cohérence renforce la confiance dans un mouvement haussier durable. "
                conclusion += "Stratégie recommandée: positions longues avec gestion de risque adaptée à l'horizon de trading."
            elif direction == "baissière":
                conclusion += "Cette cohérence renforce la confiance dans un mouvement baissier durable. "
                conclusion += "Stratégie recommandée: positions courtes avec gestion de risque adaptée à l'horizon de trading."
            else:
                conclusion += "Cette cohérence suggère une période de consolidation ou d'indécision du marché. "
                conclusion += "Stratégie recommandée: attente d'une rupture de range ou trading de la range."
        else:
            # Signaux divergents
            conclusion += "Les signaux divergent selon les horizons temporels. "
            
            # Détailler les différences
            if short_term and short_term in explanations:
                st_direction = explanations[short_term]["direction"]
                conclusion += f"À court terme, la tendance est {st_direction}. "
            
            if mid_term and mid_term in explanations:
                mt_direction = explanations[mid_term]["direction"]
                conclusion += f"À moyen terme, la tendance est {mt_direction}. "
            
            if long_term and long_term in explanations:
                lt_direction = explanations[long_term]["direction"]
                conclusion += f"À long terme, la tendance est {lt_direction}. "
            
            conclusion += "Stratégie recommandée: adapter les décisions de trading à l'horizon visé, "
            conclusion += "avec une prudence accrue due aux signaux contradictoires."
        
        # Ajouter une note sur la volatilité
        volatilities = []
        for key in [short_term, mid_term, long_term]:
            if key and key in explanations:
                volatilities.append(explanations[key]["volatility"])
        
        if "élevée" in volatilities:
            conclusion += " Note: La volatilité attendue est élevée, ce qui nécessite des stop-loss plus larges et une gestion rigoureuse du risque."
        elif "faible" in volatilities:
            conclusion += " Note: La volatilité attendue est faible, ce qui peut réduire le potentiel de profit à court terme."
        
        return conclusion
    
    def _visualize_trade_opportunity(self, data: pd.DataFrame, opportunity: Dict) -> Dict:
        """
        Génère des visualisations pour une opportunité de trading
        
        Args:
            data: DataFrame avec les données OHLCV
            opportunity: Opportunité de trading
            
        Returns:
            Chemins des visualisations générées
        """
        symbol = opportunity.get("symbol", "unknown")
        side = opportunity.get("side", "")
        entry_price = opportunity.get("entry_price", 0)
        stop_loss = opportunity.get("stop_loss", 0)
        take_profit = opportunity.get("take_profit", 0)
        score = opportunity.get("score", 0)
        
        # 1. Créer un graphique récapitulatif de l'opportunité
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # Tracer les prix récents
        recent_data = data.tail(30)
        ax.plot(recent_data.index, recent_data['close'], label="Prix de clôture", color='blue')
        
        # Marquer le prix d'entrée
        ax.axhline(y=entry_price, color='green', linestyle='--', label=f"Entrée: {entry_price:.2f}")
        
        # Marquer le stop loss et take profit
        ax.axhline(y=stop_loss, color='red', linestyle='--', label=f"Stop Loss: {stop_loss:.2f}")
        ax.axhline(y=take_profit, color='purple', linestyle='--', label=f"Take Profit: {take_profit:.2f}")
        
        # Marquer la direction
        if side == "BUY":
            arrow_start = entry_price * 0.99
            arrow_end = entry_price * 1.02
            arrow_color = 'green'
            arrow_label = "ACHAT"
        else:  # SELL
            arrow_start = entry_price * 1.01
            arrow_end = entry_price * 0.98
            arrow_color = 'red'
            arrow_label = "VENTE"
        
        ax.annotate('', xy=(recent_data.index[-1], arrow_end), 
                   xytext=(recent_data.index[-1], arrow_start),
                   arrowprops=dict(facecolor=arrow_color, width=2, headwidth=10))
        
        ax.text(recent_data.index[-1], arrow_end, f" {arrow_label}", 
               color=arrow_color, fontweight='bold', fontsize=12)
        
        # Ajouter les informations de l'opportunité
        info_text = (
            f"Opportunité: {side} {symbol}\n"
            f"Score: {score:.1f}/100\n"
            f"Ratio risque/récompense: {abs((take_profit-entry_price)/(entry_price-stop_loss)):.2f}\n"
        )
        
        # Ajouter les principales prédictions LSTM si disponibles
        lstm_prediction = opportunity.get("lstm_prediction", {})
        
        if lstm_prediction:
            # Trouver l'horizon court terme
            short_term = None
            for key in lstm_prediction.keys():
                if "12" in key or "short" in key:
                    short_term = lstm_prediction[key]
                    break
            
            if short_term:
                direction_prob = short_term.get("direction_probability", 50)
                momentum = short_term.get("predicted_momentum", 0)
                volatility = short_term.get("predicted_volatility", 0)
                
                info_text += (
                    f"Prédiction LSTM (court terme):\n"
                    f"Direction: {direction_prob:.1f}% de hausse\n"
                    f"Momentum: {momentum:.2f}\n"
                    f"Volatilité: {volatility:.2f}x la normale\n"
                )
        
        # Ajouter le texte d'info
        ax.text(0.02, 0.97, info_text, transform=ax.transAxes, fontsize=10,
               verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        # Configurer le graphique
        ax.set_title(f"Opportunité de Trading {side} - {symbol}", fontsize=14)
        ax.set_xlabel("Date")
        ax.set_ylabel("Prix")
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Sauvegarder le graphique
        trade_dir = os.path.join(self.output_dir, "trade_opportunities")
        os.makedirs(trade_dir, exist_ok=True)
        
        trade_chart = os.path.join(trade_dir, f"{symbol}_{side}_{datetime.now().strftime('%Y%m%d_%H%M%S')}_trade.png")
        plt.tight_layout()
        plt.savefig(trade_chart)
        plt.close(fig)
        
        # Retourner les chemins des visualisations
        return {
            "trade_summary_chart": trade_chart
        }
    
    def _generate_visualizations(self, data: pd.DataFrame,
                               featured_data: pd.DataFrame,
                               normalized_data: pd.DataFrame,
                               prediction: Dict,
                               important_factors: Dict,
                               output_prefix: Optional[str] = None) -> Dict:
        """
        Génère des visualisations explicatives pour les prédictions
        
        Args:
            data: DataFrame des données OHLCV originales
            featured_data: DataFrame avec les caractéristiques
            normalized_data: DataFrame avec les caractéristiques normalisées
            prediction: Prédiction du modèle
            important_factors: Facteurs importants identifiés
            output_prefix: Préfixe pour les noms de fichiers
            
        Returns:
            Dictionnaire des chemins vers les visualisations générées
        """
        if output_prefix is None:
            output_prefix = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Créer un sous-répertoire pour les visualisations
        viz_dir = os.path.join(self.output_dir, "visualizations")
        os.makedirs(viz_dir, exist_ok=True)
        
        visualization_paths = {}
        
        # 1. Visualisation des probabilités de direction par horizon
        direction_chart = self._visualize_direction_probabilities(prediction, output_prefix, viz_dir)
        visualization_paths["direction_probabilities"] = direction_chart
        
        # 2. Visualisation des facteurs importants
        importance_chart = self._visualize_feature_importance(important_factors, output_prefix, viz_dir)
        visualization_paths["feature_importance"] = importance_chart
        
        # 3. Visualisation des indicateurs clés
        indicators_chart = self._visualize_key_indicators(data, featured_data, prediction, output_prefix, viz_dir)
        visualization_paths["key_indicators"] = indicators_chart
        
        # 4. Visualisation SHAP si disponible
        if self.shap_explainer is not None:
            try:
                shap_chart = self._visualize_shap_values(normalized_data, output_prefix, viz_dir)
                visualization_paths["shap_values"] = shap_chart
            except Exception as e:
                logger.warning(f"Impossible de générer la visualisation SHAP: {str(e)}")
        
        return visualization_paths
    
    def _visualize_direction_probabilities(self, prediction: Dict, 
                                         output_prefix: str, 
                                         viz_dir: str) -> str:
        """
        Visualise les probabilités de direction pour chaque horizon
        
        Args:
            prediction: Prédiction du modèle
            output_prefix: Préfixe pour le nom de fichier
            viz_dir: Répertoire de sortie
            
        Returns:
            Chemin vers la visualisation générée
        """
        # Extraire les horizons et les probabilités
        horizons = []
        probabilities = []
        
        for horizon_key, horizon_prediction in prediction.items():
            # Extraire l'horizon (comme "3h", "12h", etc.)
            horizon_name = horizon_key.replace("horizon_", "").replace("_", " ").upper() if "_" in horizon_key else horizon_key
            
            # Extraire la probabilité de direction
            direction_prob = horizon_prediction.get("direction_probability", 50)
            
            horizons.append(horizon_name)
            probabilities.append(direction_prob)
        
        # Si aucun horizon trouvé, retourner None
        if not horizons:
            return None
        
        # Créer le graphique
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Tracer les probabilités comme des barres
        bar_colors = ['green' if p > 50 else 'red' for p in probabilities]
        bars = ax.bar(horizons, probabilities, color=bar_colors, alpha=0.7)
        
        # Ajouter une ligne horizontale à 50% (neutre)
        ax.axhline(y=50, color='gray', linestyle='--', alpha=0.7)
        
        # Ajouter les valeurs au-dessus des barres
        for bar, prob in zip(bars, probabilities):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 1,
                   f'{prob:.1f}%', ha='center', va='bottom')
        
        # Configurer le graphique
        ax.set_ylim([0, 105])  # Laisser de l'espace pour les annotations
        ax.set_ylabel('Probabilité de hausse (%)')
        ax.set_xlabel('Horizon de prédiction')
        ax.set_title('Probabilités de direction par horizon temporel')
        ax.grid(True, alpha=0.3)
        
        # Ajouter une légende pour l'interprétation
        ax.text(0.05, 0.05, "< 50%: Tendance baissière\n> 50%: Tendance haussière", 
               transform=ax.transAxes, fontsize=10,
               bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        # Sauvegarder le graphique
        output_path = os.path.join(viz_dir, f"{output_prefix}_direction_probabilities.png")
        plt.tight_layout()
        plt.savefig(output_path)
        plt.close(fig)
        
        return output_path
    
    def _visualize_feature_importance(self, important_factors: Dict,
                                    output_prefix: str,
                                    viz_dir: str) -> str:
        """
        Visualise l'importance des caractéristiques pour chaque horizon
        
        Args:
            important_factors: Facteurs importants par horizon
            output_prefix: Préfixe pour le nom de fichier
            viz_dir: Répertoire de sortie
            
        Returns:
            Chemin vers la visualisation générée
        """
        # S'il n'y a qu'un seul horizon, faire un simple graphique à barres
        if len(important_factors) == 1:
            horizon_key = list(important_factors.keys())[0]
            factors = important_factors[horizon_key].get("top_features", [])
            
            if not factors:
                return None
            
            # Extraire les noms et les scores
            feature_names = [f[0] for f in factors]
            importance_scores = [f[1] for f in factors]
            
            # Créer le graphique
            fig, ax = plt.subplots(figsize=(12, 6))
            
            # Tracer l'importance comme des barres
            y_pos = range(len(feature_names))
            bars = ax.barh(y_pos, importance_scores, color='skyblue')
            
            # Ajouter les noms des caractéristiques
            ax.set_yticks(y_pos)
            ax.set_yticklabels(feature_names)
            
            # Ajouter les valeurs à droite des barres
            for i, v in enumerate(importance_scores):
                ax.text(v + 0.01, i, f'{v:.3f}', va='center')
            
            # Configurer le graphique
            ax.set_xlabel('Importance')
            ax.set_title(f'Importance des caractéristiques - Horizon {horizon_key}')
            ax.grid(True, alpha=0.3)
            
            # Sauvegarder le graphique
            output_path = os.path.join(viz_dir, f"{output_prefix}_feature_importance.png")
            plt.tight_layout()
            plt.savefig(output_path)
            plt.close(fig)
            
            return output_path
        
        # Sinon, créer un graphique avec plusieurs sous-graphiques (un par horizon)
        else:
            # Calculer le nombre de lignes et colonnes pour les sous-graphiques
            n_horizons = len(important_factors)
            cols = min(n_horizons, 3)
            rows = (n_horizons + cols - 1) // cols
            
            fig, axs = plt.subplots(rows, cols, figsize=(15, 5 * rows))
            
            # Aplatir axs si nécessaire
            if rows == 1 and cols == 1:
                axs = [axs]
            elif rows == 1 or cols == 1:
                axs = axs.flatten()
            
            # Tracer chaque horizon
            for i, (horizon_key, factors_dict) in enumerate(important_factors.items()):
                if i >= len(axs):
                    break
                    
                factors = factors_dict.get("top_features", [])
                
                if not factors:
                    continue
                
                # Extraire les noms et les scores
                feature_names = [f[0] for f in factors]
                importance_scores = [f[1] for f in factors]
                
                # Sous-graphique actuel
                if rows > 1 and cols > 1:
                    ax = axs[i // cols, i % cols]
                else:
                    ax = axs[i]
                
                # Tracer l'importance comme des barres
                y_pos = range(len(feature_names))
                bars = ax.barh(y_pos, importance_scores, color='skyblue')
                
                # Ajouter les noms des caractéristiques
                ax.set_yticks(y_pos)
                ax.set_yticklabels(feature_names)
                
                # Ajouter les valeurs à droite des barres
                for j, v in enumerate(importance_scores):
                    ax.text(v + 0.01, j, f'{v:.3f}', va='center')
                
                # Configurer le sous-graphique
                ax.set_xlabel('Importance')
                ax.set_title(f'Horizon {horizon_key}')
                ax.grid(True, alpha=0.3)
            
            # Cacher les sous-graphiques inutilisés
            for i in range(len(important_factors), rows * cols):
                if rows > 1 and cols > 1:
                    axs[i // cols, i % cols].set_visible(False)
                else:
                    axs[i].set_visible(False)
            
            # Configurer le graphique global
            fig.suptitle('Importance des caractéristiques par horizon', fontsize=16)
            plt.tight_layout(rect=[0, 0, 1, 0.96])  # Laisser de l'espace pour le titre
            
            # Sauvegarder le graphique
            output_path = os.path.join(viz_dir, f"{output_prefix}_feature_importance_multi.png")
            plt.savefig(output_path)
            plt.close(fig)
            
            return output_path
    
    def _visualize_key_indicators(self, data: pd.DataFrame,
                                featured_data: pd.DataFrame,
                                prediction: Dict,
                                output_prefix: str,
                                viz_dir: str) -> str:
        """
        Visualise les indicateurs clés avec les prédictions
        
        Args:
            data: DataFrame des données OHLCV originales
            featured_data: DataFrame avec les caractéristiques
            prediction: Prédiction du modèle
            output_prefix: Préfixe pour le nom de fichier
            viz_dir: Répertoire de sortie
            
        Returns:
            Chemin vers la visualisation générée
        """
        # Récupérer les données récentes
        recent_data = data.tail(30).copy()
        
        # Calculer le court terme horizon
        short_term_horizon = None
        for key in prediction:
            if "12" in key or "short" in key:
                short_term_horizon = key
                break
        
        if not short_term_horizon:
            return None
        
        # Récupérer les prédictions de direction pour l'horizon court terme
        short_term_prediction = prediction[short_term_horizon]
        direction_prob = short_term_prediction.get("direction_probability", 50)
        
        # Créer un graphique avec plusieurs panneaux
        fig, axs = plt.subplots(4, 1, figsize=(12, 16), gridspec_kw={'height_ratios': [3, 1, 1, 1]})
        
        # 1. Graphique des prix avec MA
        ax1 = axs[0]
        ax1.plot(recent_data.index, recent_data['close'], label='Prix', color='blue')
        
        # Ajouter les moyennes mobiles si disponibles
        if 'ema_9' in featured_data.columns:
            recent_ema9 = featured_data['ema_9'].tail(30)
            ax1.plot(recent_data.index, recent_ema9, label='EMA 9', color='green', linestyle='--')
        
        if 'ema_21' in featured_data.columns:
            recent_ema21 = featured_data['ema_21'].tail(30)
            ax1.plot(recent_data.index, recent_ema21, label='EMA 21', color='red', linestyle='--')
        
        # Ajouter une annotation pour la prédiction
        last_date = recent_data.index[-1]
        last_price = recent_data['close'].iloc[-1]
        
        # Calculer les estimations de prix
        price_change = self._estimate_price_change(
            direction_prob=direction_prob,
            momentum=short_term_prediction.get("predicted_momentum", 0),
            volatility=short_term_prediction.get("predicted_volatility", 1),
            horizon=short_term_horizon
        )
        
        target_price = last_price * (1 + price_change/100)
        
        # Ajouter une flèche pour la direction prédite
        if direction_prob > 60:  # Hausse prévue
            arrow_color = 'green'
            arrow_start = last_price
            arrow_end = target_price
            text_pos = target_price * 1.01
            
            # Ajouter un texte explicatif
            text = f"Prédiction: Hausse\n{direction_prob:.1f}% de confiance\nObjectif: {target_price:.2f} (+{price_change:.2f}%)"
            
        elif direction_prob < 40:  # Baisse prévue
            arrow_color = 'red'
            arrow_start = last_price
            arrow_end = target_price
            text_pos = target_price * 0.99
            
            # Ajouter un texte explicatif
            text = f"Prédiction: Baisse\n{(100-direction_prob):.1f}% de confiance\nObjectif: {target_price:.2f} ({price_change:.2f}%)"
            
        else:  # Neutre
            arrow_color = 'gray'
            arrow_start = last_price
            arrow_end = last_price
            text_pos = last_price * 1.01
            
            # Ajouter un texte explicatif
            text = f"Prédiction: Neutre\n{direction_prob:.1f}% / {(100-direction_prob):.1f}%\nObjectif: {target_price:.2f} ({price_change:.2f}%)"
        
        # Tracer la flèche
        ax1.annotate('', xy=(last_date + pd.Timedelta(days=1), arrow_end), 
                   xytext=(last_date, arrow_start),
                   arrowprops=dict(facecolor=arrow_color, width=2, headwidth=10))
        
        # Ajouter le texte explicatif
        ax1.text(last_date + pd.Timedelta(days=1), text_pos, text, 
               color=arrow_color, fontweight='bold', fontsize=10)
        
        ax1.set_title('Prix et Prédiction de Direction')
        ax1.set_ylabel('Prix')
        ax1.grid(True, alpha=0.3)
        ax1.legend()
        
        # 2. RSI
        ax2 = axs[1]
        if 'rsi' in featured_data.columns:
            recent_rsi = featured_data['rsi'].tail(30)
            ax2.plot(recent_data.index, recent_rsi, color='purple')
            ax2.axhline(y=70, color='red', linestyle='--', alpha=0.5)
            ax2.axhline(y=30, color='green', linestyle='--', alpha=0.5)
            ax2.set_ylim([0, 100])
            ax2.set_ylabel('RSI')
            ax2.grid(True, alpha=0.3)
        
        # 3. MACD
        ax3 = axs[2]
        if 'macd' in featured_data.columns and 'macd_signal' in featured_data.columns:
            recent_macd = featured_data['macd'].tail(30)
            recent_signal = featured_data['macd_signal'].tail(30)
            recent_hist = featured_data['macd_hist'].tail(30) if 'macd_hist' in featured_data.columns else None
            
            ax3.plot(recent_data.index, recent_macd, color='blue', label='MACD')
            ax3.plot(recent_data.index, recent_signal, color='red', label='Signal')
            
            if recent_hist is not None:
                # Colorer les histogrammes en fonction du signe
                for i in range(len(recent_hist)):
                    color = 'green' if recent_hist.iloc[i] > 0 else 'red'
                    ax3.bar(recent_data.index[i], recent_hist.iloc[i], color=color, alpha=0.5, width=0.7)
            
            ax3.set_ylabel('MACD')
            ax3.grid(True, alpha=0.3)
            ax3.legend()
        
        # 4. Volume
        ax4 = axs[3]
        volume_colors = ['green' if recent_data['close'].iloc[i] > recent_data['open'].iloc[i] else 'red' 
                         for i in range(len(recent_data))]
        ax4.bar(recent_data.index, recent_data['volume'], color=volume_colors, alpha=0.7, width=0.7)
        ax4.set_ylabel('Volume')
        ax4.grid(True, alpha=0.3)
        
        # Configurer le graphique global
        plt.tight_layout()
        
        # Sauvegarder le graphique
        output_path = os.path.join(viz_dir, f"{output_prefix}_key_indicators.png")
        plt.savefig(output_path)
        plt.close(fig)
        
        return output_path
    
    def _visualize_shap_values(self, normalized_data: pd.DataFrame,
                             output_prefix: str,
                             viz_dir: str) -> str:
        """
        Visualise les valeurs SHAP pour le modèle LSTM
        
        Args:
            normalized_data: DataFrame des données normalisées
            output_prefix: Préfixe pour le nom de fichier
            viz_dir: Répertoire de sortie
            
        Returns:
            Chemin vers la visualisation générée
        """
        if self.shap_explainer is None:
            return None
        
        try:
            # Préparer les données pour SHAP
            X = self._prepare_data_for_shap(normalized_data)
            
            # Calculer les valeurs SHAP
            shap_values = self.shap_explainer.shap_values(X)
            
            # Utiliser les noms de caractéristiques
            feature_names = normalized_data.columns.tolist()
            
            # Créer la figure pour le résumé SHAP
            plt.figure(figsize=(12, 8))
            
            # Afficher le résumé SHAP (pour la première sortie si plusieurs)
            if isinstance(shap_values, list):
                # Première sortie (généralement la direction)
                shap.summary_plot(shap_values[0], X, feature_names=feature_names, show=False)
            else:
                shap.summary_plot(shap_values, X, feature_names=feature_names, show=False)
            
            # Sauvegarder le graphique
            output_path = os.path.join(viz_dir, f"{output_prefix}_shap_summary.png")
            plt.tight_layout()
            plt.savefig(output_path)
            plt.close()
            
            return output_path
            
        except Exception as e:
            logger.error(f"Erreur lors de la visualisation SHAP: {str(e)}")
            import traceback
            traceback.print_exc()
            return None
            
    def _explain_trade_opportunity(self, opportunity: Dict) -> str:
        """
        Génère une explication textuelle pour une opportunité de trading
        
        Args:
            opportunity: Opportunité de trading
            
        Returns:
            Explication textuelle
        """
        # Extraire les informations clés
        symbol = opportunity.get("symbol", "")
        side = opportunity.get("side", "")
        entry_price = opportunity.get("entry_price", 0)
        stop_loss = opportunity.get("stop_loss", 0)
        take_profit = opportunity.get("take_profit", 0)
        score = opportunity.get("score", 0)
        
        # Calculer le risque et le reward
        if side == "BUY":
            risk_pct = abs((entry_price - stop_loss) / entry_price * 100)
            reward_pct = abs((take_profit - entry_price) / entry_price * 100)
        else:  # SELL
            risk_pct = abs((stop_loss - entry_price) / entry_price * 100)
            reward_pct = abs((entry_price - take_profit) / entry_price * 100)
        
        risk_reward_ratio = reward_pct / risk_pct if risk_pct > 0 else 0
        
        # Obtenir les signaux techniques et LSTM
        technical_signals = opportunity.get("signals", {}).get("signals", [])
        lstm_confidence = opportunity.get("lstm_confidence", {})
        
        # Déterminer le niveau de confiance
        confidence_level = "élevée"
        if score < 65:
            confidence_level = "faible"
        elif score < 80:
            confidence_level = "modérée"
        
        # Construire l'explication
        explanation = (
            f"Opportunité de trading {side} sur {symbol} identifiée avec un score de {score}/100 "
            f"(confiance {confidence_level}). "
        )
        
        # Explication de l'entrée
        explanation += (
            f"Le prix d'entrée suggéré est de {entry_price:.4f}, avec un stop-loss à {stop_loss:.4f} "
            f"({risk_pct:.2f}% de risque) et un take-profit à {take_profit:.4f} "
            f"({reward_pct:.2f}% de gain potentiel). "
            f"Cela donne un ratio risque/récompense de {risk_reward_ratio:.2f}. "
        )
        
        # Analyse technique
        if technical_signals:
            explanation += "Les signaux techniques favorables incluent: " + ", ".join(technical_signals[:3])
            if len(technical_signals) > 3:
                explanation += f" et {len(technical_signals) - 3} autres signaux"
            explanation += ". "
        
        # Prédictions LSTM
        if lstm_confidence:
            alignment = lstm_confidence.get("direction_alignment", 0)
            momentum = lstm_confidence.get("momentum_alignment", 0)
            
            # Déterminer l'alignement des prédictions LSTM
            if alignment > 30:
                explanation += "Les prédictions LSTM confirment fortement cette direction, "
            elif alignment > 10:
                explanation += "Les prédictions LSTM confirment modérément cette direction, "
            elif alignment > -10:
                explanation += "Les prédictions LSTM sont neutres, "
            else:
                explanation += "Attention: les prédictions LSTM divergent de l'analyse technique, "
                
            # Commenter le momentum
            if momentum > 20:
                explanation += "avec un fort momentum favorable. "
            elif momentum > 5:
                explanation += "avec un momentum positif. "
            elif momentum > -5:
                explanation += "avec un momentum neutre. "
            else:
                explanation += "avec un momentum défavorable qui suggère la prudence. "
        
        # Recommandation finale
        if score >= 80:
            explanation += (
                f"Cette opportunité présente un excellent potentiel et mérite une position de taille standard "
                f"avec un ratio risque/récompense attractif de {risk_reward_ratio:.2f}."
            )
        elif score >= 70:
            explanation += (
                f"Cette opportunité est solide et mérite une position de taille modérée "
                f"avec un bon ratio risque/récompense de {risk_reward_ratio:.2f}."
            )
        else:
            explanation += (
                f"Cette opportunité présente un potentiel intéressant mais comporte des risques accrus. "
                f"Une position de taille réduite est recommandée si vous décidez de la prendre."
            )
        
        return explanation

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Explainer pour le modèle LSTM")
    parser.add_argument("--model", type=str, required=True, help="Chemin vers le modèle LSTM")
    parser.add_argument("--data", type=str, required=True, help="Chemin vers les données OHLCV")
    parser.add_argument("--symbol", type=str, default="BTCUSDT", help="Symbole à analyser")
    
    args = parser.parse_args()
    
    # Initialiser l'explainer
    explainer = ModelExplainer(model_path=args.model)
    
    # Charger les données
    try:
        data = pd.read_csv(args.data, parse_dates=['timestamp'])
        data.set_index('timestamp', inplace=True)
        
        # Obtenir une prédiction du modèle
        featured_data, normalized_data = explainer._prepare_data(data)
        X, _ = explainer.feature_engineering.create_multi_horizon_data(
            normalized_data,
            sequence_length=explainer.model.input_length,
            horizons=explainer.model.horizon_periods,
            is_training=False
        )
        
        # Faire une prédiction avec le modèle
        predictions = explainer.model.predict(data)
        
        # Expliquer la prédiction
        explanation = explainer.explain_prediction(
            data=data,
            prediction=predictions,
            generate_visualizations=True,
            output_prefix=args.symbol
        )
        
        print("Explication générée avec succès. Voir les visualisations dans le répertoire de sortie.")
        
    except Exception as e:
        print(f"Erreur: {str(e)}")