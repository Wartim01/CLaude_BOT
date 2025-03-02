# ai/models/lstm_model.py
"""
Modèle LSTM avancé avec mécanismes d'attention pour la prédiction des mouvements de marché
Intègre un mécanisme d'attention multi-tête inspiré des architectures Transformer
"""
import os
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import (
    Input, LSTM, Dense, Dropout, BatchNormalization, 
    Bidirectional, Concatenate, Add, Attention, TimeDistributed,
    Conv1D, MaxPooling1D, GlobalAveragePooling1D, Multiply
)
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l1_l2
import tensorflow.keras.backend as K
from typing import Dict, List, Tuple, Union, Optional
from datetime import datetime

from config.config import DATA_DIR
from utils.logger import setup_logger
from ai.models.attention import MultiHeadAttention, TemporalAttentionBlock, TimeSeriesAttention

logger = setup_logger("lstm_model")

class LSTMModel:
    """
    Modèle LSTM avancé avec mécanismes d'attention pour prédire les mouvements de marché
    
    Caractéristiques:
    - Architecture LSTM bidirectionnelle avec mécanisme d'attention multi-tête
    - Prédictions multi-horizon (court, moyen, long terme)
    - Prédictions multi-facteur (direction, volatilité, momentum, volume)
    - Connexions résiduelles pour une meilleure propagation du gradient
    - Mécanismes de régularisation avancés (dropout spatial, batch normalization)
    """
    def __init__(self, 
                 input_length: int = 60,
                 feature_dim: int = 30,
                 lstm_units: List[int] = [128, 64, 32],
                 dropout_rate: float = 0.3,
                 learning_rate: float = 0.001,
                 l1_reg: float = 0.0001,
                 l2_reg: float = 0.0001,
                 use_attention: bool = True,
                 attention_type: str = 'multi_head',  # 'simple', 'multi_head', 'temporal_block'
                 use_residual: bool = True,
                 prediction_horizons: List[int] = [12, 24, 96]):  # 3h, 6h, 24h (avec des bougies de 15min)
        """
        Initialise le modèle LSTM
        
        Args:
            input_length: Nombre de pas de temps en entrée
            feature_dim: Dimension des caractéristiques d'entrée
            lstm_units: Liste des unités LSTM pour chaque couche
            dropout_rate: Taux de dropout pour la régularisation
            learning_rate: Taux d'apprentissage du modèle
            l1_reg: Régularisation L1
            l2_reg: Régularisation L2
            use_attention: Utiliser les mécanismes d'attention
            attention_type: Type d'attention à utiliser ('simple', 'multi_head', 'temporal_block')
            use_residual: Utiliser les connexions résiduelles
            prediction_horizons: Horizons de prédiction en nombre de périodes
        """
        self.input_length = input_length
        self.feature_dim = feature_dim
        self.lstm_units = lstm_units
        self.dropout_rate = dropout_rate
        self.learning_rate = learning_rate
        self.l1_reg = l1_reg
        self.l2_reg = l2_reg
        self.use_attention = use_attention
        self.attention_type = attention_type
        self.use_residual = use_residual
        self.prediction_horizons = prediction_horizons
        
        # Identifier les horizons court/moyen/long terme
        self.short_term_idx = 0  # Premier horizon (le plus court)
        self.mid_term_idx = 1 if len(prediction_horizons) > 1 else 0  # Moyen terme ou court terme si un seul horizon
        self.long_term_idx = -1  # Dernier horizon (le plus long)
        
        # Nombre de sorties pour chaque horizon (direction, volatilité, volume, momentum)
        self.output_dim = 4
        
        # Variables pour mémoriser la normalisation
        self.feature_means = None
        self.feature_stds = None
        
        # Créer le modèle Keras
        self.model = self._build_model()
        self.model_path = os.path.join(DATA_DIR, "models", "production", "lstm_model.h5")
        
        # Métriques de performance du modèle
        self.metrics_history = []
        
    def _build_model(self) -> Model:
        """
        Construit le modèle LSTM avec attention et connexions résiduelles
        
        Returns:
            Modèle Keras compilé
        """
        # Entrée de forme (batch_size, sequence_length, num_features)
        inputs = Input(shape=(self.input_length, self.feature_dim), name="market_sequence")
        
        # Couche de convolution 1D pour extraire des motifs locaux (comme un feature extractor)
        conv_branch = Conv1D(filters=64, kernel_size=3, padding='same', activation='relu', 
                            kernel_regularizer=l1_l2(l1=self.l1_reg, l2=self.l2_reg))(inputs)
        conv_branch = BatchNormalization()(conv_branch)
        conv_branch = MaxPooling1D(pool_size=2, padding='same')(conv_branch)
        
        # Branch principale avec LSTM bidirectionnel
        x = Bidirectional(LSTM(self.lstm_units[0], return_sequences=True, 
                              kernel_regularizer=l1_l2(l1=self.l1_reg, l2=self.l2_reg)))(inputs)
        x = BatchNormalization()(x)
        x = Dropout(self.dropout_rate)(x)
        
        # Couches LSTM intermédiaires avec connexions résiduelles
        for i, units in enumerate(self.lstm_units[1:], 1):
            # Couche LSTM bidirectionnelle
            lstm_out = Bidirectional(LSTM(units, return_sequences=True, 
                                         kernel_regularizer=l1_l2(l1=self.l1_reg, l2=self.l2_reg)))(x)
            lstm_out = BatchNormalization()(lstm_out)
            
            # Connexion résiduelle si les dimensions correspondent
            if self.use_residual and 2*units == 2*self.lstm_units[i-1]:  # Facteur 2 pour bidirectionnel
                x = Add()([x, lstm_out])
            else:
                # Projection pour faire correspondre les dimensions si nécessaire
                x = Dense(2*units, activation='relu')(x)
                x = Add()([x, lstm_out])
                
            x = Dropout(self.dropout_rate)(x)
        
        # Fusion de la branche convolutive avec la branche LSTM
        # (nécessite d'adapter les dimensions pour la fusion)
        conv_branch = TimeDistributed(Dense(2*self.lstm_units[-1]))(conv_branch)
        
        # Adapter les dimensions temporelles si nécessaire (à cause du MaxPooling)
        if conv_branch.shape[1] != x.shape[1]:
            # Utilisation de couches de suréchantillonnage (upsampling) pour faire correspondre les dimensions
            upsampling_factor = x.shape[1] // conv_branch.shape[1]
            
            # Utiliser la répétition temporelle pour l'upsampling
            expanded_conv = tf.keras.layers.Lambda(
                lambda t: tf.repeat(t, repeats=upsampling_factor, axis=1)
            )(conv_branch)
            
            # Ajuster la forme finale si nécessaire
            target_shape = x.shape[1:]
            if expanded_conv.shape[1:] != target_shape:
                expanded_conv = tf.keras.layers.Reshape(target_shape)(expanded_conv)
                
            conv_branch = expanded_conv
        
        # Fusionner les branches
        combined = Add()([x, conv_branch])
        
        # Mécanisme d'attention avancé
        if self.use_attention:
            if self.attention_type == 'simple':
                # Attention simple sur la séquence temporelle
                attention_layer = Attention()([combined, combined])
                
                # Porte d'attention pour donner plus d'importance aux parties pertinentes de la séquence
                attention_weights = Dense(1, activation='sigmoid')(attention_layer)
                attention_weights = tf.keras.layers.Reshape((self.input_length, 1))(attention_weights)
                
                # Appliquer l'attention
                attended = Multiply()([combined, attention_weights])
                x = Add()([combined, attended])  # Connexion résiduelle avec l'attention
                
            elif self.attention_type == 'multi_head':
                # Attention multi-tête
                multi_head_attention = MultiHeadAttention(
                    num_heads=8,
                    head_dim=32,
                    dropout=self.dropout_rate
                )(combined)
                
                # Connexion résiduelle
                x = Add()([combined, multi_head_attention])
                x = BatchNormalization()(x)
                
            elif self.attention_type == 'temporal_block':
                # Bloc d'attention temporelle complet (inspiré des Transformers)
                temporal_attention = TemporalAttentionBlock(
                    num_heads=8,
                    head_dim=32,
                    ff_dim=128,
                    dropout=self.dropout_rate
                )(combined)
                
                x = temporal_attention
                
            else:
                # Attention spécifique aux séries temporelles financières
                time_series_attention = TimeSeriesAttention(
                    filters=64,
                    kernel_size=3
                )(combined)
                
                x = time_series_attention
        else:
            x = combined
        
        # Créer une sortie pour chaque horizon de prédiction
        outputs = []
        
        for horizon in self.prediction_horizons:
            # Pour chaque horizon, prédire:
            # 1. Direction du prix (probabilité de hausse, baisse)
            # 2. Volatilité attendue
            # 3. Volume relatif attendu
            # 4. Momentum (force de la tendance)
            
            # Récupérer le contexte global à partir de la séquence
            global_context = GlobalAveragePooling1D()(x)
            
            # Couche dense finale pour chaque horizon
            horizon_output = Dense(32, activation='relu')(global_context)
            horizon_output = BatchNormalization()(horizon_output)
            horizon_output = Dropout(0.2)(horizon_output)
            
            # Sorties multiples pour chaque horizon
            direction = Dense(1, activation='sigmoid', name=f'direction_h{horizon}')(horizon_output)  # Probabilité de hausse
            volatility = Dense(1, activation='relu', name=f'volatility_h{horizon}')(horizon_output)  # Volatilité relative
            volume = Dense(1, activation='relu', name=f'volume_h{horizon}')(horizon_output)  # Volume relatif
            momentum = Dense(1, activation='tanh', name=f'momentum_h{horizon}')(horizon_output)  # Momentum (-1 à 1)
            
            outputs.extend([direction, volatility, volume, momentum])
        
        # Créer le modèle final
        model = Model(inputs=inputs, outputs=outputs)
        
        # Compilation avec des métriques adaptées
        losses = []
        metrics = []
        
        for i in range(len(self.prediction_horizons)):
            # Binary crossentropy pour la direction
            losses.append('binary_crossentropy')
            metrics.append('accuracy')
            
            # MSE pour volatilité, volume et momentum
            for _ in range(3):
                losses.append('mse')
                metrics.append('mae')
        
        model.compile(
            optimizer=Adam(learning_rate=self.learning_rate),
            loss=losses,
            metrics=metrics
        )
        
        return model
    
    def preprocess_data(self, data: pd.DataFrame, is_training: bool = True) -> Tuple[np.ndarray, List[np.ndarray]]:
        """
        Prétraite les données pour l'entraînement ou la prédiction
        
        Args:
            data: DataFrame avec les données OHLCV et indicateurs
            is_training: Indique si le prétraitement est pour l'entraînement
            
        Returns:
            X: Données d'entrée normalisées
            Y: Liste des données cibles pour chaque sortie (vide si is_training=False)
        """
        # Vérifier que nous avons suffisamment de données
        if len(data) < self.input_length + max(self.prediction_horizons):
            raise ValueError(f"Données insuffisantes. Nécessite au moins {self.input_length + max(self.prediction_horizons)} points.")
        
        # Sélectionner les colonnes de caractéristiques (toutes sauf la date)
        feature_columns = [col for col in data.columns if col != 'timestamp' and col != 'date']
        
        if self.feature_dim != len(feature_columns):
            logger.warning(f"Dimension des caractéristiques configurée ({self.feature_dim}) ne correspond pas au nombre réel de colonnes ({len(feature_columns)})")
            self.feature_dim = len(feature_columns)
        
        # Normalisation des caractéristiques
        if is_training or self.feature_means is None:
            # Calculer les moyennes et écarts-types sur les données d'entraînement
            self.feature_means = data[feature_columns].mean()
            self.feature_stds = data[feature_columns].std()
            self.feature_stds = self.feature_stds.replace(0, 1)  # Éviter la division par zéro
        
        # Normaliser les données
        normalized_data = (data[feature_columns] - self.feature_means) / self.feature_stds
        
        # Créer les séquences d'entrée
        X = []
        Y_direction = []
        Y_volatility = []
        Y_volume = []
        Y_momentum = []
        
        # Pour chaque horizon de prédiction
        Y_outputs = [[] for _ in range(len(self.prediction_horizons) * 4)]  # 4 sorties par horizon
        
        for i in range(len(normalized_data) - self.input_length - max(self.prediction_horizons)):
            # Séquence d'entrée
            X.append(normalized_data.iloc[i:i+self.input_length].values)
            
            if is_training:
                # Pour chaque horizon, préparer les cibles
                for h_idx, horizon in enumerate(self.prediction_horizons):
                    current_price = data['close'].iloc[i+self.input_length-1]
                    future_price = data['close'].iloc[i+self.input_length+horizon-1]
                    
                    # Direction (1 si hausse, 0 si baisse)
                    direction = 1 if future_price > current_price else 0
                    Y_outputs[h_idx*4].append(direction)
                    
                    # Volatilité (écart-type normalisé des prix sur l'horizon)
                    future_prices = data['close'].iloc[i+self.input_length:i+self.input_length+horizon]
                    volatility = future_prices.pct_change().std() * np.sqrt(horizon)  # Annualisé
                    Y_outputs[h_idx*4+1].append(volatility)
                    
                    # Volume (volume moyen relatif sur l'horizon)
                    current_volume = data['volume'].iloc[i+self.input_length-1]
                    future_volumes = data['volume'].iloc[i+self.input_length:i+self.input_length+horizon]
                    relative_volume = future_volumes.mean() / current_volume if current_volume > 0 else 1.0
                    Y_outputs[h_idx*4+2].append(relative_volume)
                    
                    # Momentum (importance du mouvement relatif)
                    price_change_pct = (future_price - current_price) / current_price
                    # Normaliser entre -1 et 1 avec tanh
                    momentum = np.tanh(price_change_pct * 5)  # Facteur 5 pour amplifier les petits mouvements
                    Y_outputs[h_idx*4+3].append(momentum)
        
        # Convertir en tableaux numpy
        X = np.array(X)
        Y = [np.array(y) for y in Y_outputs] if is_training else []
        
        return X, Y
    
    def train(self, train_data: pd.DataFrame, validation_data: pd.DataFrame = None, 
             epochs: int = 100, batch_size: int = 32, patience: int = 20,
             save_best: bool = True) -> Dict:
        """
        Entraîne le modèle LSTM
        
        Args:
            train_data: DataFrame avec les données d'entraînement
            validation_data: DataFrame avec les données de validation
            epochs: Nombre d'époques d'entraînement
            batch_size: Taille du batch
            patience: Patience pour l'early stopping
            save_best: Sauvegarder le meilleur modèle
            
        Returns:
            Historique d'entraînement
        """
        # Préparer les données d'entraînement
        X_train, Y_train = self.preprocess_data(train_data, is_training=True)
        
        # Préparer les données de validation si fournies
        validation_data_keras = None
        if validation_data is not None:
            X_val, Y_val = self.preprocess_data(validation_data, is_training=True)
            validation_data_keras = (X_val, Y_val)
        
        # Créer les répertoires pour sauvegarder le modèle si nécessaire
        if save_best:
            model_dir = os.path.dirname(self.model_path)
            if not os.path.exists(model_dir):
                os.makedirs(model_dir, exist_ok=True)
        
        # Callbacks pour l'entraînement
        callbacks = [
            EarlyStopping(
                monitor='val_loss' if validation_data is not None else 'loss',
                patience=patience,
                restore_best_weights=True
            ),
            ReduceLROnPlateau(
                monitor='val_loss' if validation_data is not None else 'loss',
                factor=0.5,
                patience=int(patience/2),
                min_lr=1e-6
            )
        ]
        
        if save_best:
            callbacks.append(
                ModelCheckpoint(
                    filepath=self.model_path,
                    monitor='val_loss' if validation_data is not None else 'loss',
                    save_best_only=True,
                    save_weights_only=False
                )
            )
        
        # Entraîner le modèle
        history = self.model.fit(
            x=X_train,
            y=Y_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_data=validation_data_keras,
            callbacks=callbacks,
            verbose=1
        )
        
        # Sauvegarder les métriques
        self._save_metrics(history.history)
        
        return history.history
    
    def predict(self, data: pd.DataFrame) -> Dict:
        """
        Fait des prédictions avec le modèle entraîné
        
        Args:
            data: DataFrame avec les données récentes
            
        Returns:
            Dictionnaire avec les prédictions pour chaque horizon
        """
        # Prétraiter les données
        X, _ = self.preprocess_data(data, is_training=False)
        
        # Faire la prédiction
        predictions = self.model.predict(X)
        
        # Reformater les prédictions
        results = {}
        
        # Dernière séquence pour la prédiction la plus récente
        latest_prediction_idx = -1
        
        # Pour chaque horizon
        for h_idx, horizon in enumerate(self.prediction_horizons):
            horizon_name = f"horizon_{horizon}"
            
            # Indice de base pour les 4 sorties de cet horizon
            base_idx = h_idx * 4
            
            # Récupérer les prédictions pour cet horizon
            direction = float(predictions[base_idx][latest_prediction_idx][0])
            volatility = float(predictions[base_idx+1][latest_prediction_idx][0])
            volume = float(predictions[base_idx+2][latest_prediction_idx][0])
            momentum = float(predictions[base_idx+3][latest_prediction_idx][0])
            
            results[horizon_name] = {
                "direction_probability": direction,  # Probabilité de hausse
                "predicted_volatility": volatility,
                "predicted_relative_volume": volume,
                "predicted_momentum": momentum,
                "prediction_timestamp": datetime.now().isoformat()
            }
        
        return results
    
    def evaluate(self, test_data: pd.DataFrame) -> Dict:
        """
        Évalue le modèle sur des données de test
        
        Args:
            test_data: DataFrame avec les données de test
            
        Returns:
            Métriques d'évaluation
        """
        # Prétraiter les données
        X_test, Y_test = self.preprocess_data(test_data, is_training=True)
        
        # Évaluer le modèle
        evaluation = self.model.evaluate(X_test, Y_test, verbose=1)
        
        # Organiser les résultats
        results = {}
        metric_names = []
        
        # Construire les noms des métriques
        for h_idx, horizon in enumerate(self.prediction_horizons):
            for output in ["direction", "volatility", "volume", "momentum"]:
                metric_base = f"h{horizon}_{output}"
                metric_names.append(f"{metric_base}_loss")
                
                # Ajouter l'accuracy pour la direction, MAE pour les autres
                if output == "direction":
                    metric_names.append(f"{metric_base}_accuracy")
                else:
                    metric_names.append(f"{metric_base}_mae")
        
        # Remplir les résultats
        results["loss"] = evaluation[0]
        
        for i, metric_name in enumerate(metric_names):
            results[metric_name] = evaluation[i+1]
        
        # Calcul des métriques additionnelles
        # Faire des prédictions
        predictions = self.model.predict(X_test)
        
        # Pour chaque horizon, calculer des métriques spécifiques
        for h_idx, horizon in enumerate(self.prediction_horizons):
            base_idx = h_idx * 4
            
            # Direction (classificaion binaire)
            y_true_direction = Y_test[base_idx]
            y_pred_direction = (predictions[base_idx] > 0.5).astype(int).flatten()
            
            # Calcul de précision, rappel, F1
            true_positives = np.sum((y_pred_direction == 1) & (y_true_direction == 1))
            false_positives = np.sum((y_pred_direction == 1) & (y_true_direction == 0))
            false_negatives = np.sum((y_pred_direction == 0) & (y_true_direction == 1))
            
            precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
            recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
            
            results[f"h{horizon}_direction_precision"] = precision
            results[f"h{horizon}_direction_recall"] = recall
            results[f"h{horizon}_direction_f1"] = f1
        
        return results
    
    def save(self, path: Optional[str] = None) -> None:
        """
        Sauvegarde le modèle et ses paramètres
        
        Args:
            path: Chemin pour sauvegarder le modèle (utilise self.model_path par défaut)
        """
        save_path = path or self.model_path
        
        # Créer le répertoire si nécessaire
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        
        # Sauvegarder le modèle
        self.model.save(save_path)
        
        # Sauvegarder les paramètres de normalisation et configuration
        params = {
            "feature_means": self.feature_means.to_dict() if self.feature_means is not None else None,
            "feature_stds": self.feature_stds.to_dict() if self.feature_stds is not None else None,
            "input_length": self.input_length,
            "feature_dim": self.feature_dim,
            "lstm_units": self.lstm_units,
            "dropout_rate": self.dropout_rate,
            "learning_rate": self.learning_rate,
            "l1_reg": self.l1_reg,
            "l2_reg": self.l2_reg,
            "use_attention": self.use_attention,
            "attention_type": self.attention_type,
            "use_residual": self.use_residual,
            "prediction_horizons": self.prediction_horizons,
            "last_updated": datetime.now().isoformat()
        }
        
        # Sauvegarder en JSON
        params_path = save_path.replace(".h5", "_params.json")
        import json
        with open(params_path, 'w') as f:
            json.dump(params, f, indent=2, default=str)
        
        logger.info(f"Modèle sauvegardé: {save_path}")
        logger.info(f"Paramètres sauvegardés: {params_path}")
    
    def load(self, path: Optional[str] = None) -> None:
        """
        Charge le modèle et ses paramètres
        
        Args:
            path: Chemin du modèle à charger (utilise self.model_path par défaut)
        """
        load_path = path or self.model_path
        
        # Vérifier que le modèle existe
        if not os.path.exists(load_path):
            raise FileNotFoundError(f"Modèle non trouvé: {load_path}")
        
        # Charger le modèle
        self.model = load_model(load_path)
        
        # Charger les paramètres
        params_path = load_path.replace(".h5", "_params.json")
        if os.path.exists(params_path):
            import json
            with open(params_path, 'r') as f:
                params = json.load(f)
            
            # Restaurer les paramètres
            if "feature_means" in params and params["feature_means"]:
                self.feature_means = pd.Series(params["feature_means"])
            
            if "feature_stds" in params and params["feature_stds"]:
                self.feature_stds = pd.Series(params["feature_stds"])
            
            # Restaurer la configuration
            self.input_length = params.get("input_length", self.input_length)
            self.feature_dim = params.get("feature_dim", self.feature_dim)
            self.lstm_units = params.get("lstm_units", self.lstm_units)
            self.dropout_rate = params.get("dropout_rate", self.dropout_rate)
            self.learning_rate = params.get("learning_rate", self.learning_rate)
            self.l1_reg = params.get("l1_reg", self.l1_reg)
            self.l2_reg = params.get("l2_reg", self.l2_reg)
            self.use_attention = params.get("use_attention", self.use_attention)
            self.attention_type = params.get("attention_type", self.attention_type)
            self.use_residual = params.get("use_residual", self.use_residual)
            self.prediction_horizons = params.get("prediction_horizons", self.prediction_horizons)
        
        logger.info(f"Modèle chargé: {load_path}")
    
    def _save_metrics(self, metrics: Dict) -> None:
        """
        Sauvegarde les métriques d'entraînement ou d'évaluation
        
        Args:
            metrics: Dictionnaire de métriques
        """
        # Ajouter un timestamp
        metrics_with_timestamp = {
            "timestamp": datetime.now().isoformat(),
            **metrics
        }
        
        # Ajouter à l'historique
        self.metrics_history.append(metrics_with_timestamp)
        
        # Limiter la taille de l'historique
        if len(self.metrics_history) > 100:
            self.metrics_history = self.metrics_history[-100:]
        
        # Sauvegarder l'historique
        metrics_path = self.model_path.replace(".h5", "_metrics.json")
        import json
        with open(metrics_path, 'w') as f:
            json.dump(self.metrics_history, f, indent=2, default=str)
        
    def get_prediction_confidence(self, prediction: Dict) -> Dict:
        """
        Calcule la confiance dans les prédictions en fonction de leur cohérence
        
        Args:
            prediction: Dictionnaire de prédictions du modèle
            
        Returns:
            Confiance dans les prédictions (0-1)
        """
        confidence = {}
        
        # Vérifier la cohérence des prédictions entre les différents horizons
        directions = []
        momentums = []
        
        for horizon_name, horizon_pred in prediction.items():
            directions.append(horizon_pred["direction_probability"])
            momentums.append(horizon_pred["predicted_momentum"])
        
        # Confiance globale basée sur la cohérence des directions
        direction_std = np.std(directions)
        direction_confidence = 1.0 - min(1.0, direction_std * 2)  # Plus le std est bas, plus la confiance est élevée
        
        # Confiance basée sur l'alignement direction/momentum
        # (direction > 0.5 doit correspondre à momentum > 0)
        direction_momentum_aligned = sum(1 for d, m in zip(directions, momentums) if (d > 0.5 and m > 0) or (d < 0.5 and m < 0)) / len(directions)
        
        # Confiance globale (moyenne des deux mesures)
        overall_confidence = (direction_confidence + direction_momentum_aligned) / 2
        
        # Confiances spécifiques pour chaque horizon
        for i, horizon_name in enumerate(prediction.keys()):
            horizon_confidence = {
                "overall": overall_confidence,
                "direction_confidence": direction_confidence,
                "momentum_alignment": direction_momentum_aligned,
                "strength": abs(momentums[i])  # Force du signal basée sur l'amplitude du momentum
            }
            confidence[horizon_name] = horizon_confidence
        
        return confidence
    
    def update_incrementally(self, new_data: pd.DataFrame, epochs: int = 5, 
                          learning_rate: float = 0.0005) -> Dict:
        """
        Met à jour le modèle de manière incrémentale avec de nouvelles données
        
        Args:
            new_data: Nouvelles données pour la mise à jour
            epochs: Nombre d'époques pour la mise à jour
            learning_rate: Taux d'apprentissage pour la mise à jour
            
        Returns:
            Historique de la mise à jour
        """
        # Sauvegarder les poids actuels pour pouvoir revenir en arrière si nécessaire
        original_weights = self.model.get_weights()
        
        # Réduire le taux d'apprentissage pour la mise à jour incrémentale
        K.set_value(self.model.optimizer.learning_rate, learning_rate)
        
        # Préparer les données
        X, Y = self.preprocess_data(new_data, is_training=True)
        
        # Si les données sont insuffisantes, ne pas mettre à jour
        if len(X) < 10:
            logger.warning("Données insuffisantes pour la mise à jour incrémentale")
            return {"success": False, "message": "Données insuffisantes"}
        
        # Mise à jour incrémentale
        try:
            history = self.model.fit(
                x=X,
                y=Y,
                epochs=epochs,
                batch_size=16,  # Batch size plus petit pour la mise à jour
                verbose=1
            )
            
            # Vérifier les performances après mise à jour
            # En cas de dégradation significative, revenir aux poids précédents
            evaluation_before = self.model.evaluate(X, Y, verbose=0)[0]  # Loss globale
            
            # Évaluer le modèle mis à jour sur les mêmes données
            self.model.set_weights(original_weights)
            evaluation_after = self.model.evaluate(X, Y, verbose=0)[0]  # Loss globale
            
            # Si les performances se dégradent significativement, revenir en arrière
            if evaluation_after > evaluation_before * 1.1:  # Dégradation de plus de 10%
                logger.warning("Dégradation des performances après mise à jour incrémentale. Retour aux poids précédents.")
                return {"success": False, "message": "Dégradation des performances"}
            
            # Sinon, appliquer à nouveau la mise à jour (puisqu'on est revenu aux poids d'origine)
            history = self.model.fit(
                x=X,
                y=Y,
                epochs=epochs,
                batch_size=16,
                verbose=1
            )
            
            # Sauvegarder le modèle mis à jour
            self.save()
            
            return {"success": True, "history": history.history}
            
        except Exception as e:
            # En cas d'erreur, revenir aux poids précédents
            self.model.set_weights(original_weights)
            logger.error(f"Erreur lors de la mise à jour incrémentale: {str(e)}")
            return {"success": False, "message": str(e)}
    
    def get_feature_importance(self, data: pd.DataFrame) -> Dict:
        """
        Calcule l'importance des caractéristiques à l'aide de méthodes d'explicabilité
        
        Args:
            data: DataFrame avec les données de test
            
        Returns:
            Importance des caractéristiques
        """
        # Préparer les données
        X, _ = self.preprocess_data(data, is_training=False)
        
        if len(X) == 0:
            logger.warning("Données insuffisantes pour calculer l'importance des caractéristiques")
            return {}
        
        # Obtenir les noms des caractéristiques
        feature_columns = [col for col in data.columns if col != 'timestamp' and col != 'date']
        
        # Méthode d'analyse de sensibilité pour l'importance des caractéristiques
        # Pour chaque caractéristique, calculer la variation de la sortie quand on la perturbe
        importances = {}
        
        for i, feature in enumerate(feature_columns):
            # Copier les données d'entrée
            X_perturbed = X.copy()
            
            # Perturber la caractéristique (remplacer par la moyenne)
            X_perturbed[:, :, i] = 0
            
            # Prédictions de base
            y_base = self.model.predict(X)
            
            # Prédictions avec perturbation
            y_perturbed = self.model.predict(X_perturbed)
            
            # Calculer la différence moyenne pour chaque sortie
            diff_sum = 0
            for j in range(len(y_base)):
                diff_sum += np.mean(np.abs(y_base[j] - y_perturbed[j]))
            
            # Normaliser la différence
            importance = diff_sum / len(y_base)
            
            importances[feature] = float(importance)
        
        # Normaliser les importances pour qu'elles somment à 1
        total_importance = sum(importances.values())
        if total_importance > 0:
            importances = {k: v / total_importance for k, v in importances.items()}
        
        # Trier par importance décroissante
        importances = dict(sorted(importances.items(), key=lambda item: item[1], reverse=True))
        
        return importances