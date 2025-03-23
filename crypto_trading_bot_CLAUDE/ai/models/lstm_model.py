# ai/models/lstm_model.py
"""
Architecture LSTM avancée pour prédictions multi-horizon et multi-facteur
Intègre attention, connexions résiduelles et apprentissage par transfert
"""
import os
import numpy as np  # Correction effectuée ici: "import numpy as np" au lieu de "import numpy np"
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Model, load_model, clone_model # type: ignore
from tensorflow.keras.layers import ( # type: ignore
    Input, LSTM, Dense, Dropout, BatchNormalization, 
    Bidirectional, Concatenate, Add, Multiply, Reshape,
    Conv1D, MaxPooling1D, GlobalAveragePooling1D, Lambda,
    Layer, Activation
)
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau # type: ignore
from tensorflow.keras.optimizers import Adam # type: ignore
from tensorflow.keras.regularizers import l1_l2 # type: ignore
import tensorflow.keras.backend as K # type: ignore
from typing import Dict, List, Tuple, Union, Optional
from datetime import datetime
import json

from config.config import DATA_DIR
from utils.logger import setup_logger
from ai.models.attention import MultiHeadAttention, TemporalAttentionBlock, TimeSeriesAttention

logger = setup_logger("enhanced_lstm_model")

class SpatialDropout1D(Dropout):
    """
    Dropout spatial qui supprime des canaux de caractéristiques entiers plutôt que des valeurs individuelles
    """
    def __init__(self, rate, **kwargs):
        super(SpatialDropout1D, self).__init__(rate, **kwargs)
        self.input_spec = None
    
    def call(self, inputs, training=None):
        if training is None:
            training = K.learning_phase()
        if 0. < self.rate < 1.:
            input_shape = tf.shape(inputs)
            noise_shape = tf.stack([input_shape[0], tf.constant(1), input_shape[2]])
            def dropped():
                return tf.nn.dropout(inputs, rate=self.rate, noise_shape=noise_shape)
            return tf.cond(tf.cast(training, tf.bool), lambda: dropped(), lambda: inputs)
        return inputs

class ResidualBlock(Layer):
    """
    Bloc résiduel personnalisé qui combine LSTM et connexions résiduelles
    """
    def __init__(self, units, dropout_rate=0.3, use_batch_norm=True, **kwargs):
        super(ResidualBlock, self).__init__(**kwargs)
        self.units = units
        self.dropout_rate = dropout_rate
        self.use_batch_norm = use_batch_norm
        
        # Définir les couches
        self.lstm = Bidirectional(LSTM(units, return_sequences=True, 
                                      kernel_regularizer=l1_l2(l1=1e-5, l2=1e-4)))
        self.dropout = SpatialDropout1D(dropout_rate)
        
        if use_batch_norm:
            self.batch_norm = BatchNormalization()
        
        self.projection = None
        
    def build(self, input_shape):
        # Projection pour faire correspondre les dimensions si nécessaire
        input_dim = input_shape[-1]
        output_dim = self.units * 2  # Bidirectionnel double la dimension
        
        if input_dim != output_dim:
            self.projection = Dense(output_dim)
            
        super(ResidualBlock, self).build(input_shape)
    
    def call(self, inputs, training=None):
        x = self.lstm(inputs)
        
        if self.use_batch_norm:
            x = self.batch_norm(x, training=training)
            
        x = self.dropout(x, training=training)
        
        # Connexion résiduelle
        if self.projection is not None:
            residual = self.projection(inputs)
        else:
            residual = inputs
            
        return Add()([x, residual])
    
    def get_config(self):
        config = super(ResidualBlock, self).get_config()
        config.update({
            'units': self.units,
            'dropout_rate': self.dropout_rate,
            'use_batch_norm': self.use_batch_norm
        })
        return config

import os
import json
from pathlib import Path

def load_best_hyperparameters(symbol="BTCUSDT", timeframe="15m"):
    """
    Load the best hyperparameters from optimization results
    
    Args:
        symbol: Trading pair symbol
        timeframe: Timeframe for the model
        
    Returns:
        Dictionary with best parameters or empty dict if not found
    """
    # Path to optimization results
    from config.config import DATA_DIR
    optimization_dir = os.path.join(DATA_DIR, "models", "optimization")
    
    if not os.path.exists(optimization_dir):
        logger.warning(f"Optimization directory not found: {optimization_dir}")
        return {}
    
    # Find the most recent best_params file
    param_files = [f for f in os.listdir(optimization_dir) 
                  if f.startswith("best_params_") and f.endswith(".json")]
    
    if not param_files:
        logger.warning("No optimization results found")
        return {}
    
    # Sort by date (newest first)
    param_files.sort(reverse=True)
    best_params_path = os.path.join(optimization_dir, param_files[0])
    
    try:
        with open(best_params_path, 'r') as f:
            params_data = json.load(f)
            
        best_params = params_data.get("best_params", {})
        if best_params:
            logger.info(f"Loaded best hyperparameters from {best_params_path}")
            return best_params
        else:
            return {}
            
    except Exception as e:
        logger.error(f"Error loading best hyperparameters: {str(e)}")
        return {}

# Import the model parameters
from config.model_params import LSTM_DEFAULT_PARAMS, LSTM_OPTIMIZED_PARAMS

class LSTMModel(Model):
    """
    Modèle LSTM avancé pour prédictions multi-horizon et multi-facteur.
    Intègre éventuellement attention et connexions résiduelles.
    """
    def __init__(self, input_length=None, feature_dim=None, lstm_units=None, dropout_rate=None, learning_rate=None, *args, **kwargs):
        # Extract missing parameters from kwargs with default values
        timeframe = kwargs.get("timeframe", "15m")
        symbol = kwargs.get("symbol", "BTCUSDT")
        l1_reg = kwargs.get("l1_reg", None)
        l2_reg = kwargs.get("l2_reg", None)
        use_optimized_params = kwargs.get("use_optimized_params", False)
        use_attention = kwargs.get("use_attention", None)
        use_residual = kwargs.get("use_residual", None)
        prediction_horizons = kwargs.get("prediction_horizons", [(12, "3h", True), (48, "12h", True), (192, "48h", True)])
        
        self.input_length = input_length if input_length is not None else 60
        self.feature_dim = feature_dim if feature_dim is not None else 66
        self.lstm_units = lstm_units or [128, 64, 32]
        self.dropout_rate = dropout_rate
        self.learning_rate = learning_rate
        self.l1_reg = l1_reg
        self.l2_reg = l2_reg
        self.use_attention = use_attention
        self.use_residual = use_residual
        self.prediction_horizons = prediction_horizons
        
        # First check if there are optimized parameters for this timeframe
        optimized_params = None
        if use_optimized_params and timeframe in LSTM_OPTIMIZED_PARAMS:
            # Only use if they've been optimized (not the default values)
            if LSTM_OPTIMIZED_PARAMS[timeframe]["last_optimized"] is not None:
                optimized_params = LSTM_OPTIMIZED_PARAMS[timeframe]
                logger.info(f"Using optimized parameters for {timeframe} from config.model_params")
                logger.info(f"Last optimization: {optimized_params['last_optimized']} with F1 score: {optimized_params['f1_score']:.4f}")
        
        # Load parameters with priority: 
        # 1. Explicitly provided params
        # 2. Optimized params from config
        # 3. Default params from config
        # 4. Hyperparameter optimization results (as before)
        
        # Start with default params
        params_source = LSTM_DEFAULT_PARAMS
        
        # Override with optimized params if available
        if optimized_params is not None:
            params_source = optimized_params
        
        # Set parameters, with explicitly provided params taking precedence
        self.input_length = input_length if input_length is not None else params_source["sequence_length"]
        self.feature_dim = feature_dim  # Feature dimension updated to 78
        from config.model_params import LSTM_OPTIMIZED_PARAMS as latest_params
        if use_optimized_params and timeframe in latest_params and latest_params[timeframe].get("last_optimized") is not None:
            optimized_params = latest_params[timeframe]
            self.input_length = optimized_params.get("sequence_length", self.input_length)
            self.lstm_units = optimized_params.get("lstm_units", self.lstm_units)
            self.dropout_rate = optimized_params.get("dropout_rate", self.dropout_rate)
            self.learning_rate = optimized_params.get("learning_rate", self.learning_rate)
            self.l1_reg = optimized_params.get("l1_regularization", self.l1_reg)
            self.l2_reg = optimized_params.get("l2_regularization", self.l2_reg)
            logger.info(f"Reloaded optimized parameters for {timeframe} from config: sequence_length={self.input_length}, lstm_units={self.lstm_units}")
        self.prediction_horizons = prediction_horizons
        
        if use_optimized_params and optimized_params is None:
            best_params = load_best_hyperparameters(symbol, timeframe)
            
            if best_params:
                logger.info("Using optimized hyperparameters from saved files:")
                
                # Extract LSTM units from optimization results
                if "lstm_units_first" in best_params and "lstm_layers" in best_params:
                    lstm_units_list = [best_params["lstm_units_first"]]
                    for i in range(1, best_params["lstm_layers"]):
                        lstm_units_list.append(lstm_units_list[-1] // 2)
                    self.lstm_units = lstm_units_list
                    logger.info(f"  - LSTM units: {lstm_units_list}")
                
                # Map other parameters
                if "sequence_length" in best_params:
                    self.input_length = best_params["sequence_length"]
                    logger.info(f"  - Input length: {self.input_length}")
                
                if "dropout_rate" in best_params:
                    self.dropout_rate = best_params["dropout_rate"]
                    logger.info(f"  - Dropout rate: {self.dropout_rate}")
                
                if "learning_rate" in best_params:
                    self.learning_rate = best_params["learning_rate"]
                    logger.info(f"  - Learning rate: {self.learning_rate}")
                
                if "l1_reg" in best_params:
                    self.l1_reg = best_params["l1_reg"]
                    logger.info(f"  - L1 regularization: {self.l1_reg}")
                
                if "l2_reg" in best_params:
                    self.l2_reg = best_params["l2_reg"]
                    logger.info(f"  - L2 regularization: {self.l2_reg}")
                    logger.info(f"  - Dropout rate: {self.dropout_rate}")
                
                if "learning_rate" in best_params:
                    self.learning_rate = best_params["learning_rate"]
                    logger.info(f"  - Learning rate: {self.learning_rate}")
                
                if "l1_reg" in best_params:
                    self.l1_reg = best_params["l1_reg"]
                    logger.info(f"  - L1 regularization: {self.l1_reg}")
                
                if "l2_reg" in best_params:
                    self.l2_reg = best_params["l2_reg"]
                    logger.info(f"  - L2 regularization: {self.l2_reg}")
        
        # Finally, log which parameters are being used
        logger.info(f"Model parameters: sequence_length={self.input_length}, lstm_units={self.lstm_units}, "
                   f"dropout={self.dropout_rate}, learning_rate={self.learning_rate}, "
                   f"l1_reg={self.l1_reg}, l2_reg={self.l2_reg}")
                    
        self.model = None
        
        # Store symbol and timeframe for potential retraining
        self.symbol = symbol
        self.timeframe = timeframe
        
        # Build the model with the parameters
        self.build_model()

    def get_config(self):
        config = super(LSTMModel, self).get_config()
        config.update({
            "input_length": self.input_length,
            "feature_dim": self.feature_dim,
            "lstm_units": self.lstm_units,
            "dropout_rate": self.dropout_rate,
            "learning_rate": self.learning_rate,
            "l1_reg": self.l1_reg,
            "l2_reg": self.l2_reg,
            "use_attention": self.use_attention,
            "use_residual": self.use_residual,
            "prediction_horizons": self.prediction_horizons
        })
        return config

    def build_model(self):
        """
        Construit le modèle LSTM multi-sorties complet
        """
        from tensorflow.keras.layers import Input, LSTM, Dense, Dropout, Bidirectional # type: ignore
        from tensorflow.keras.models import Model # type: ignore
        from tensorflow.keras.optimizers import Adam # type: ignore
        from tensorflow.keras.regularizers import l1_l2 # type: ignore
        
        inputs = Input(shape=(self.input_length, self.feature_dim))
        x = inputs

        # Première couche LSTM bidirectionnelle
        if len(self.lstm_units) > 0:
            x = Bidirectional(LSTM(self.lstm_units[0], return_sequences=len(self.lstm_units) > 1))(x)
            x = SpatialDropout1D(self.dropout_rate)(x)
            
            # Couches LSTM intermédiaires
            for i in range(1, len(self.lstm_units)-1):
                x = Bidirectional(LSTM(self.lstm_units[i], return_sequences=True))(x)
                x = SpatialDropout1D(self.dropout_rate)(x)
            
            # Dernière couche LSTM
            if len(self.lstm_units) > 1:
                lstm_out = Bidirectional(LSTM(self.lstm_units[-1], return_sequences=False))(x)
            else:
                lstm_out = x
        else:
            lstm_out = Bidirectional(LSTM(64, return_sequences=False))(x)
        
        # Application de dropout
        x = Dropout(self.dropout_rate)(lstm_out)
        
        # Couche dense avec régularisation
        x = Dense(64, activation='relu', kernel_regularizer=l1_l2(l1=self.l1_reg, l2=self.l2_reg))(x)
        
        # Sorties pour chaque horizon
        outputs = []
        for horizon in self.prediction_horizons:
            direction = Dense(1, activation='sigmoid', name=f'direction_{horizon}')(x)
            outputs.append(direction)
        
        self.model = Model(inputs=inputs, outputs=outputs)
        
        # Fix: Use a list for metrics instead of a string
        self.model.compile(
            optimizer=Adam(learning_rate=self.learning_rate),
            loss=['binary_crossentropy' for _ in outputs],
            metrics=['accuracy']  # Use a list instead of a string
        )
        return self.model

    def build_single_output_model(self, horizon_idx=0):
        """
        Construit un modèle LSTM avec une seule sortie pour l'optimisation
        
        Args:
            horizon_idx: Indice de l'horizon à utiliser (0 = court terme)
            
        Returns:
            Modèle Keras avec une seule sortie
        """
        from tensorflow.keras.layers import Input, LSTM, Dense, Dropout, Bidirectional # type: ignore
        from tensorflow.keras.models import Model # type: ignore
        from tensorflow.keras.optimizers import Adam # type: ignore
        from tensorflow.keras.regularizers import l1_l2 # type: ignore
        
        inputs = Input(shape=(self.input_length, self.feature_dim))
        x = inputs

        # Première couche LSTM bidirectionnelle
        if len(self.lstm_units) > 0:
            x = Bidirectional(LSTM(self.lstm_units[0], return_sequences=len(self.lstm_units) > 1))(x)
            x = Dropout(self.dropout_rate)(x)
            
            # Couches LSTM intermédiaires
            for i in range(1, len(self.lstm_units)-1):
                x = Bidirectional(LSTM(self.lstm_units[i], return_sequences=True))(x)
                x = Dropout(self.dropout_rate)(x)
            
            # Dernière couche LSTM
            if len(self.lstm_units) > 1:
                x = Bidirectional(LSTM(self.lstm_units[-1], return_sequences=False))(x)
            else:
                x = Bidirectional(LSTM(self.lstm_units[0], return_sequences=False))(x)
        else:
            x = Bidirectional(LSTM(64, return_sequences=False))(x)
        
        # Application de dropout
        x = Dropout(self.dropout_rate)(x)
        
        # Couche dense avec régularisation
        x = Dense(64, activation='relu', kernel_regularizer=l1_l2(l1=self.l1_reg, l2=self.l2_reg))(x)
        
        # Sortie unique (direction pour l'horizon spécifié)
        horizon = self.prediction_horizons[horizon_idx] if horizon_idx < len(self.prediction_horizons) else self.prediction_horizons[0]
        output = Dense(1, activation='sigmoid', name=f'direction_{horizon}')(x)
        
        # Créer et compiler le modèle
        single_model = Model(inputs=inputs, outputs=output)
        # Fix: Use a list for metrics here too
        single_model.compile(
            optimizer=Adam(learning_rate=self.learning_rate),
            loss='binary_crossentropy',
            metrics=['accuracy']  # Use a list instead of a string
        )
        
        return single_model

    def save(self, path: str) -> None:
        self.model.save(path)
        
    def load(self, path: Optional[str] = None) -> None:
        """
        Charge le modèle depuis le disque.
        Args:
            path: Chemin du modèle (utilise le chemin par défaut si None)
        """
        import os
        from tensorflow.keras.models import load_model # type: ignore
        load_path = path or self.model_path
        if not os.path.exists(load_path):
            raise FileNotFoundError(f"Modèle non trouvé: {load_path}")
        self.model = load_model(
            load_path,
            custom_objects={
                'MultiHeadAttention': MultiHeadAttention,
                'TemporalAttentionBlock': TemporalAttentionBlock,
                'TimeSeriesAttention': TimeSeriesAttention,
                'SpatialDropout1D': SpatialDropout1D,
                'ResidualBlock': ResidualBlock,
                'LSTMModel': LSTMModel
            }
        )
        logger.info(f"Modèle principal chargé: {load_path}")

class EnhancedLSTMModel:
    """
    Modèle LSTM avancé pour prédictions multi-horizon et multi-facteur
    
    Caractéristiques:
    - Architecture LSTM bidirectionnelle avec attention multi-tête
    - Prédictions multi-horizon (court, moyen, long terme)
    - Prédictions multi-facteur (direction, volatilité, volume, momentum)
    - Connexions résiduelles pour une meilleure propagation du gradient
    - Dropout spatial pour une meilleure régularisation
    - Mécanismes d'alerte précoce pour les retournements de marché
    - Intégration avec l'apprentissage par transfert
    """
    def __init__(self, 
                 input_length: int = 60,
                 feature_dim: int = 66,  # Updated default to match harmonized features (66)
                 lstm_units: List[int] = [128, 96, 64],
                 dropout_rate: float = 0.3,  # Reduced dropout from 0.5 to 0.3
                 learning_rate: float = 0.001,  # Increased learning rate to 0.001
                 l1_reg: float = 0.0001,
                 l2_reg: float = 0.0001,
                 use_attention: bool = True,
                 attention_heads: int = 8,
                 use_residual: bool = True,
                 # Format: (périodes, nom_lisible, est_principal)
                 prediction_horizons: List[Tuple[int, str, bool]] = [
                     (12, "3h", True),    # Court terme (3h avec bougies de 15min)
                     (48, "12h", True),   # Moyen terme (12h)
                     (192, "48h", True),  # Long terme (48h)
                     (384, "96h", False)  # Très long terme (96h, optionnel)
                 ]):
        """
        Initialise le modèle LSTM avancé
        
        Args:
            input_length: Nombre de pas de temps en entrée
            feature_dim: Dimension des caractéristiques d'entrée
            lstm_units: Liste des unités LSTM pour chaque couche
            dropout_rate: Taux de dropout pour la régularisation
            learning_rate: Taux d'apprentissage du modèle
            l1_reg: Régularisation L1
            l2_reg: Régularisation L2
            use_attention: Utiliser les mécanismes d'attention
            attention_heads: Nombre de têtes d'attention
            use_residual: Utiliser les connexions résiduelles
            prediction_horizons: Horizons de prédiction avec format (périodes, nom, est_principal)
        """
        self.input_length = input_length
        self.feature_dim = feature_dim
        self.lstm_units = lstm_units
        self.dropout_rate = dropout_rate
        self.learning_rate = learning_rate
        self.l1_reg = l1_reg
        self.l2_reg = l2_reg
        self.use_attention = use_attention
        self.attention_heads = attention_heads
        self.use_residual = use_residual
        self.prediction_horizons = prediction_horizons
        
        # Extraire juste les périodes pour la compatibilité
        self.horizon_periods = [h[0] for h in prediction_horizons]
        
        # Séparer les horizons principaux et secondaires
        self.main_horizons = [h for h in prediction_horizons if h[2]]
        
        # Identifier les indices pour les différents horizons
        self.short_term_idx = 0  # Premier horizon (le plus court)
        self.mid_term_idx = min(1, len(prediction_horizons)-1)  # Second horizon ou le premier si un seul
        self.long_term_idx = min(2, len(prediction_horizons)-1)  # Troisième horizon ou le dernier disponible
        
        # Facteurs de sortie pour chaque horizon (direction, volatilité, volume, momentum)
        self.factors = ["direction", "volatility", "volume", "momentum"]
        self.num_factors = len(self.factors)
        
        # Variables pour mémoriser la normalisation
        self.scalers = {}
        
        # Créer les modèles Keras (principal et auxiliaires)
        self.model = self._build_model()
        self.reversal_detector = self._build_reversal_detector()
        
        # Chemins des modèles
        self.models_dir = os.path.join(DATA_DIR, "models", "production")
        os.makedirs(self.models_dir, exist_ok=True)
        
        self.model_path = os.path.join(self.models_dir, "enhanced_lstm_model.h5")
        self.reversal_detector_path = os.path.join(self.models_dir, "reversal_detector_model.h5")
        
        # Historique des performances du modèle
        self.metrics_history = []
        
        # Méta-informations pour l'apprentissage par transfert
        self.transfer_info = {
            "base_symbol": None,
            "trained_symbols": [],
            "transfer_history": []
        }
        
    def _build_model(self) -> Model:
        """
        Construit le modèle LSTM multi-horizon et multi-facteur
        
        Returns:
            Modèle Keras compilé
        """
        # Entrée de forme (batch_size, sequence_length, num_features)
        inputs = Input(shape=(self.input_length, self.feature_dim), name="market_sequence")
        
        # 1. Branche d'extraction de caractéristiques à court terme (convolutive)
        short_term_features = Conv1D(filters=64, kernel_size=3, padding='same', activation='relu', 
                                    kernel_regularizer=l1_l2(l1=self.l1_reg, l2=self.l2_reg),
                                    name="short_term_conv1")(inputs)
        short_term_features = BatchNormalization()(short_term_features)
        short_term_features = Conv1D(filters=32, kernel_size=3, padding='same', activation='relu',
                                    name="short_term_conv2")(short_term_features)
        short_term_features = MaxPooling1D(pool_size=2, padding='same')(short_term_features)
        
        # 2. Branche principale LSTM avec blocs résiduels
        x = inputs
        
        # Appliquer des blocs résiduels en série
        for i, units in enumerate(self.lstm_units):
            x = ResidualBlock(
                units=units,
                dropout_rate=self.dropout_rate,
                use_batch_norm=True,
                name=f"residual_block_{i+1}"
            )(x)
        
        # 3. Appliquer l'attention si activée
        if self.use_attention:
            # Attention multi-tête inspirée des transformers
            mha = MultiHeadAttention(
                num_heads=self.attention_heads,
                head_dim=32,
                dropout=self.dropout_rate,
                name="multi_head_attention"
            )(x)
            
            # Connexion résiduelle après l'attention
            x = Add(name="post_attention_residual")([x, mha])
            x = BatchNormalization(name="post_attention_norm")(x)
        
        # 4. Combiner avec les caractéristiques à court terme
        # Adapter les dimensions si nécessaire
        if short_term_features.shape[1] != x.shape[1]:
            # Utiliser un redimensionnement adaptatif
            scale_factor = x.shape[1] / short_term_features.shape[1]
            
            def resize_temporal(tensor, scale):
                # Redimensionne la dimension temporelle d'un tenseur 3D
                shape = tf.shape(tensor)
                target_length = tf.cast(tf.cast(shape[1], tf.float32) * scale, tf.int32)
                
                # Redimensionner avec un reshape et des opérations de répétition
                resized = tf.image.resize(
                    tf.expand_dims(tensor, 3),
                    [target_length, shape[2]],
                    method='nearest'
                )
                return tf.squeeze(resized, 3)
            
            short_term_features = Lambda(
                lambda t: resize_temporal(t, scale_factor),
                name="temporal_resize"
            )(short_term_features)
        
        # Projeter les caractéristiques à court terme pour correspondre à la dimension de x
        short_term_features = Dense(
            x.shape[-1], 
            activation='relu',
            name="short_term_projection"
        )(short_term_features)
        
        # Combiner par addition (connexion résiduelle)
        combined = Add(name="feature_combination")([x, short_term_features])
        
        # Couche de contexte global pour la sortie
        global_context = GlobalAveragePooling1D(name="global_pooling")(combined)
        
        # 5. Couches denses partagées pour l'extraction de caractéristiques de haut niveau
        shared_features = Dense(
            128, 
            activation='relu',
            kernel_regularizer=l1_l2(l1=self.l1_reg, l2=self.l2_reg),
            name="shared_features"
        )(global_context)
        shared_features = BatchNormalization(name="shared_features_norm")(shared_features)
        shared_features = Dropout(0.2, name="shared_features_dropout")(shared_features)
        
        # 6. Créer une sortie pour chaque horizon et chaque facteur
        outputs = []
        output_names = []
        
        for h_idx, (horizon, horizon_name, _) in enumerate(self.prediction_horizons):
            # Couche spécifique à l'horizon
            horizon_features = Dense(
                64,
                activation='relu',
                name=f"horizon_{horizon_name}_features"
            )(shared_features)
            
            # Direction (probabilité de hausse)
            direction = Dense(
                1, 
                activation='sigmoid',
                name=f"direction_{horizon_name}"
            )(horizon_features)
            outputs.append(direction)
            output_names.append(f"direction_{horizon_name}")
            
            # Volatilité (relative)
            volatility = Dense(
                1, 
                activation='relu',  # La volatilité est toujours positive
                name=f"volatility_{horizon_name}"
            )(horizon_features)
            outputs.append(volatility)
            output_names.append(f"volatility_{horizon_name}")
            
            # Volume relatif
            volume = Dense(
                1, 
                activation='relu',  # Le volume relatif est toujours positif
                name=f"volume_{horizon_name}"
            )(horizon_features)
            outputs.append(volume)
            output_names.append(f"volume_{horizon_name}")
            
            # Momentum (force de la tendance)
            momentum = Dense(
                1, 
                activation='tanh',  # Tanh pour avoir une valeur entre -1 et 1
                name=f"momentum_{horizon_name}"
            )(horizon_features)
            outputs.append(momentum)
            output_names.append(f"momentum_{horizon_name}")
        
        # 7. Créer le modèle final
        model = Model(inputs=inputs, outputs=outputs, name="multi_horizon_lstm")
        
        # 8. Compiler avec des pertes et métriques appropriées
        losses = []
        metrics = []
        
        for output_name in output_names:
            if output_name.startswith("direction"):
                # Classification binaire pour la direction
                losses.append('binary_crossentropy')
                metrics.append('accuracy')
            else:
                # Régression pour les autres facteurs
                losses.append('mse')  # Erreur quadratique moyenne
                metrics.append('mae')  # Erreur absolue moyenne
        
        # Compiler le modèle
        model.compile(
            optimizer=Adam(learning_rate=self.learning_rate),
            loss=losses,
            metrics=metrics
        )
        
        return model
    
    def _build_reversal_detector(self) -> Model:
        """
        Construit un modèle spécialisé pour détecter les retournements de marché majeurs
        
        Returns:
            Modèle de détection des retournements
        """
        # Ce modèle utilise la même entrée que le modèle principal mais se spécialise
        # dans la détection des patterns de retournement de marché
        
        # Entrée de forme (batch_size, sequence_length, num_features)
        inputs = Input(shape=(self.input_length, self.feature_dim), name="market_sequence")
        
        # Utiliser des convolutions pour détecter des motifs locaux
        x = Conv1D(filters=64, kernel_size=3, padding='same', activation='relu')(inputs)
        x = BatchNormalization()(x)
        x = Conv1D(filters=64, kernel_size=5, padding='same', activation='relu')(x)
        x = BatchNormalization()(x)
        x = MaxPooling1D(pool_size=2)(x)
        
        # LSTM bidirectionnel pour capturer les dépendances temporelles
        x = Bidirectional(LSTM(64, return_sequences=True))(x)
        x = Dropout(0.3)(x)
        
        # Attention pour se concentrer sur les parties importantes de la séquence
        attention_layer = TimeSeriesAttention(filters=64)(x)
        
        # Contexte global
        global_features = GlobalAveragePooling1D()(attention_layer)
        
        # Couches denses
        x = Dense(32, activation='relu')(global_features)
        x = BatchNormalization()(x)
        x = Dropout(0.2)(x)
        
        # Sorties: probabilité et ampleur du retournement
        reversal_probability = Dense(1, activation='sigmoid', name="reversal_probability")(x)
        reversal_magnitude = Dense(1, activation='relu', name="reversal_magnitude")(x)
        
        # Créer le modèle
        model = Model(inputs=inputs, outputs=[reversal_probability, reversal_magnitude], name="reversal_detector")
        
        # Compiler
        model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss=['binary_crossentropy', 'mse'],
            metrics=[['accuracy'], ['mae']]
        )
        
        return model
    
    def preprocess_data(self, data: pd.DataFrame, 
                       feature_engineering, is_training: bool = True) -> Tuple:
        """
        Prétraite les données pour l'entraînement ou la prédiction
        
        Args:
            data: DataFrame avec les données OHLCV et indicateurs
            feature_engineering: Instance FeatureEngineering pour créer/normaliser les caractéristiques
            is_training: Indique si le prétraitement est pour l'entraînement
            
        Returns:
            X: Données d'entrée normalisées
            y_list: Liste des données cibles pour chaque sortie (vide si is_training=False)
        """
        # 1. Créer les caractéristiques avancées
        featured_data = feature_engineering.create_features(
            data, 
            include_time_features=True,
            include_price_patterns=True
        )
        
        # 2. Normaliser les caractéristiques
        normalized_data = feature_engineering.scale_features(
            featured_data,
            is_training=is_training,
            method='standard',
            feature_group='lstm'
        )
        
        # 3. Convertir en séquences pour LSTM
        X, y_list = feature_engineering.create_multi_horizon_data(
            normalized_data,
            sequence_length=self.input_length,
            horizons=self.horizon_periods,
            is_training=is_training
        )
        
        # 4. Créer les labels pour le détecteur de retournement si en mode entraînement
        if is_training:
            y_reversal = self._create_reversal_labels(data)
            return X, y_list, y_reversal
            
        return X, y_list
    
    def _create_reversal_labels(self, data: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """
        Crée les labels pour le détecteur de retournement
        
        Args:
            data: DataFrame avec les données OHLCV
            
        Returns:
            Tuple de (probabilité de retournement, amplitude du retournement)
        """
        # Calcul des rendements
        returns = data['close'].pct_change()
        
        # Identifier les retournements majeurs (mouvements brusques après une tendance)
        reversal_probability = []
        reversal_magnitude = []
        
        # Fenêtre pour la détection de retournements
        window = 20
        threshold = 0.03  # 3% de mouvement pour un retournement significatif
        
        for i in range(window, len(data) - self.input_length - max(self.horizon_periods)):
            # Tendance précédente (sur la fenêtre)
            previous_returns = returns.iloc[i-window:i]
            previous_trend = previous_returns.mean() * window  # Rendement cumulé
            
            # Mouvement futur (sur l'horizon le plus court)
            future_price_change = (data['close'].iloc[i+self.input_length+self.horizon_periods[0]-1] - 
                                 data['close'].iloc[i+self.input_length-1]) / data['close'].iloc[i+self.input_length-1]
            
            # Un retournement est un mouvement dans la direction opposée à la tendance précédente
            is_reversal = (previous_trend * future_price_change < 0) and (abs(future_price_change) > threshold)
            
            reversal_probability.append(1.0 if is_reversal else 0.0)
            reversal_magnitude.append(abs(future_price_change))
        
        return np.array(reversal_probability), np.array(reversal_magnitude)
    
    def train(self, train_data: pd.DataFrame, feature_engineering, 
             validation_data: pd.DataFrame = None, 
             epochs: int = 100, batch_size: int = 32, 
             patience: int = 20, save_best: bool = True,
             symbol: str = None) -> Dict:
        """
        Entraîne le modèle LSTM multi-horizon
        
        Args:
            train_data: DataFrame avec les données d'entraînement
            feature_engineering: Instance FeatureEngineering
            validation_data: DataFrame avec les données de validation
            epochs: Nombre d'époques d'entraînement
            batch_size: Taille du batch
            patience: Patience pour l'early stopping
            save_best: Sauvegarder le meilleur modèle
            symbol: Symbole de la paire de trading
            
        Returns:
            Historique d'entraînement
        """
        # Prétraiter les données d'entraînement
        X_train, y_train, y_reversal_train = self.preprocess_data(
            train_data, 
            feature_engineering,
            is_training=True
        )
        
        # Prétraiter les données de validation si fournies
        validation_data_main = None
        validation_data_reversal = None
        
        if validation_data is not None:
            X_val, y_val, y_reversal_val = self.preprocess_data(
                validation_data,
                feature_engineering,
                is_training=True
            )
            validation_data_main = (X_val, y_val)
            validation_data_reversal = (X_val, y_reversal_val)
        
        # Callbacks pour l'entraînement du modèle principal
        callbacks_main = [
            EarlyStopping(
                monitor='val_loss' if validation_data is not None else 'loss',
                patience=patience,
                restore_best_weights=True
            ),
            ReduceLROnPlateau(
                monitor='val_loss' if validation_data is not None else 'loss',
                factor=0.5,
                patience=patience // 2,
                min_lr=1e-6
            )
        ]
        
        if save_best:
            callbacks_main.append(
                ModelCheckpoint(
                    filepath=self.model_path,
                    monitor='val_loss' if validation_data is not None else 'loss',
                    save_best_only=True,
                    save_weights_only=False
                )
            )
        
        # 1. Entraîner le modèle principal
        logger.info("Entraînement du modèle LSTM multi-horizon...")
        history_main = self.model.fit(
            x=X_train,
            y=y_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_data=validation_data_main,
            callbacks=callbacks_main,
            verbose=1
        )
        
        # 2. Entraîner le détecteur de retournement
        logger.info("Entraînement du détecteur de retournement...")
        history_reversal = self.reversal_detector.fit(
            x=X_train,
            y=y_reversal_train,
            epochs=max(30, epochs//2),  # Moins d'époques pour ce modèle plus simple
            batch_size=batch_size,
            validation_data=validation_data_reversal,
            callbacks=[
                EarlyStopping(patience=patience//2, restore_best_weights=True),
                ReduceLROnPlateau(factor=0.5, patience=patience//4, min_lr=1e-6)
            ],
            verbose=1
        )
        
        if save_best:
            self.reversal_detector.save(self.reversal_detector_path)
        
        # Sauvegarder les métriques
        self._save_metrics(history_main.history, symbol)
        
        # Si c'est un nouveau symbole, mettre à jour les métadonnées de transfert
        if symbol and symbol not in self.transfer_info['trained_symbols']:
            self.transfer_info['trained_symbols'].append(symbol)
            
            # Si c'est le premier symbole, le définir comme symbole de base
            if self.transfer_info['base_symbol'] is None:
                self.transfer_info['base_symbol'] = symbol
                
            # Enregistrer cette session d'entraînement
            self.transfer_info['transfer_history'].append({
                'symbol': symbol,
                'timestamp': datetime.now().isoformat(),
                'epochs': len(history_main.history['loss']),
                'final_loss': float(history_main.history['loss'][-1])
            })
            
            # Sauvegarder les métadonnées de transfert
            self._save_transfer_info()
        
        return {
            'main_model': history_main.history,
            'reversal_detector': history_reversal.history
        }
    
    def predict(self, data: pd.DataFrame, feature_engineering) -> Dict:
        """
        Fait des prédictions avec le modèle entraîné
        
        Args:
            data: DataFrame avec les données récentes
            feature_engineering: Instance FeatureEngineering
            
        Returns:
            Dictionnaire avec les prédictions pour chaque horizon et facteur
            et l'alerte de retournement
        """
        # Prétraiter les données
        X, _ = self.preprocess_data(data, feature_engineering, is_training=False)
        
        # Faire les prédictions avec le modèle principal
        predictions = self.model.predict(X)
        
        # Faire les prédictions avec le détecteur de retournement
        reversal_prob, reversal_mag = self.reversal_detector.predict(X)
        
        # Organiser les résultats
        results = {}
        
        # Dernière séquence pour la prédiction la plus récente
        latest_idx = -1
        
        # Pour chaque horizon
        for h_idx, (horizon, horizon_name, _) in enumerate(self.prediction_horizons):
            # Indice de base pour les 4 facteurs de cet horizon
            base_idx = h_idx * self.num_factors
            
            # Prédictions pour cet horizon
            direction = float(predictions[base_idx][latest_idx][0])
            volatility = float(predictions[base_idx+1][latest_idx][0])
            volume = float(predictions[base_idx+2][latest_idx][0])
            momentum = float(predictions[base_idx+3][latest_idx][0])
            
            # Convertir la prédiction de direction en probabilité (0-100%)
            direction_probability = direction * 100
            
            # Déterminer la tendance prédite
            trend = "HAUSSIER" if direction > 0.5 else "BAISSIER"
            trend_strength = abs(direction - 0.5) * 2  # Force de 0 à 1
            
            # Confiance dans la prédiction
            confidence = trend_strength * (1 - volatility)  # Plus la volatilité est faible, plus la confiance est élevée
            
            results[horizon_name] = {
                "direction": trend,
                "direction_probability": direction_probability,
                "trend_strength": float(trend_strength),
                "predicted_volatility": float(volatility),
                "predicted_volume": float(volume),
                "predicted_momentum": float(momentum),
                "confidence": float(confidence),
                "prediction_timestamp": datetime.now().isoformat()
            }
        
        # Ajouter les prédictions de retournement
        results["reversal_alert"] = {
            "probability": float(reversal_prob[latest_idx][0]),
            "magnitude": float(reversal_mag[latest_idx][0]),
            "is_warning": reversal_prob[latest_idx][0] > 0.7,  # Alerte si probabilité > 70%
            "prediction_timestamp": datetime.now().isoformat()
        }
        
        return results
    
    def save(self, path: Optional[str] = None) -> None:
        """
        Sauvegarde le modèle sur disque
        
        Args:
            path: Chemin de sauvegarde (utilise le chemin par défaut si None)
        """
        save_path = path or self.model_path
        
        # Créer le répertoire si nécessaire
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        
        # Sauvegarder le modèle principal
        self.model.save(save_path)
        logger.info(f"Modèle principal sauvegardé: {save_path}")
        
        # Sauvegarder le détecteur de retournement
        reversal_path = os.path.splitext(save_path)[0] + "_reversal.h5"
        self.reversal_detector.save(reversal_path)
        logger.info(f"Détecteur de retournement sauvegardé: {reversal_path}")
    
    def load(self, path: Optional[str] = None) -> None:
        """
        Charge le modèle depuis le disque
        
        Args:
            path: Chemin du modèle (utilise le chemin par défaut si None)
        """
        load_path = path or self.model_path
        
        # Vérifier si le fichier existe
        if not os.path.exists(load_path):
            raise FileNotFoundError(f"Modèle non trouvé: {load_path}")
        
        # Charger le modèle principal
        self.model = load_model(
            load_path,
            custom_objects={
                'MultiHeadAttention': MultiHeadAttention,
                'TemporalAttentionBlock': TemporalAttentionBlock,
                'TimeSeriesAttention': TimeSeriesAttention,
                'SpatialDropout1D': SpatialDropout1D,
                'ResidualBlock': ResidualBlock
            }
        )
        logger.info(f"Modèle principal chargé: {load_path}")
        
        # Essayer de charger le détecteur de retournement
        reversal_path = os.path.splitext(load_path)[0] + "_reversal.h5"
        if os.path.exists(reversal_path):
            self.reversal_detector = load_model(
                reversal_path,
                custom_objects={
                    'TimeSeriesAttention': TimeSeriesAttention
                }
            )
            logger.info(f"Détecteur de retournement chargé: {reversal_path}")
        else:
            logger.warning(f"Détecteur de retournement non trouvé: {reversal_path}")
    
    def _save_metrics(self, history: Dict, symbol: Optional[str] = None) -> None:
        """
        Sauvegarde les métriques d'entraînement
        
        Args:
            history: Historique d'entraînement
            symbol: Symbole de la paire de trading
        """
        metrics_entry = {
            "timestamp": datetime.now().isoformat(),
            "symbol": symbol,
            "metrics": history,
            "parameters": {
                "lstm_units": self.lstm_units,
                "dropout_rate": self.dropout_rate,
                "learning_rate": self.learning_rate,
                "use_attention": self.use_attention,
                "attention_heads": self.attention_heads,
                "use_residual": self.use_residual
            }
        }
        
        self.metrics_history.append(metrics_entry)
        
        # Sauvegarder dans un fichier
        metrics_path = os.path.join(self.models_dir, "metrics_history.json")
        
        try:
            # Charger l'historique existant si disponible
            existing_metrics = []
            if os.path.exists(metrics_path):
                with open(metrics_path, 'r') as f:
                    existing_metrics = json.load(f)
            
            # Ajouter la nouvelle entrée
            existing_metrics.append(metrics_entry)
            
            # Sauvegarder
            with open(metrics_path, 'w') as f:
                json.dump(existing_metrics, f, indent=2, default=str)
                
            logger.info(f"Métriques sauvegardées: {metrics_path}")
        except Exception as e:
            logger.error(f"Erreur lors de la sauvegarde des métriques: {str(e)}")
    
    def _save_transfer_info(self) -> None:
        """Sauvegarde les métadonnées d'apprentissage par transfert"""
        transfer_path = os.path.join(self.models_dir, "transfer_learning_info.json")
        
        try:
            with open(transfer_path, 'w') as f:
                json.dump(self.transfer_info, f, indent=2, default=str)
            logger.info(f"Métadonnées de transfert sauvegardées: {transfer_path}")
        except Exception as e:
            logger.error(f"Erreur lors de la sauvegarde des métadonnées de transfert: {str(e)}")
    
    def transfer_to_new_symbol(self, symbol: str) -> None:
        """
        Prépare le modèle pour l'apprentissage par transfert sur un nouveau symbole
        
        Args:
            symbol: Nouveau symbole pour l'apprentissage par transfert
        """
        # Sauvegarder les poids du modèle source
        source_weights = self.model.get_weights()
        
        # Réduire le taux d'apprentissage pour le transfert
        K.set_value(self.model.optimizer.learning_rate, self.learning_rate * 0.5)
        
        logger.info(f"Modèle préparé pour l'apprentissage par transfert vers {symbol}")
        logger.info(f"Taux d'apprentissage réduit à {K.get_value(self.model.optimizer.learning_rate)}")
    
    def get_reversal_threshold(self, data: pd.DataFrame, feature_engineering, 
                              percentile: float = 90) -> float:
        """
        Calcule un seuil dynamique pour les alertes de retournement
        
        Args:
            data: Données historiques récentes
            feature_engineering: Instance FeatureEngineering
            percentile: Percentile pour le seuil (défaut: 90e percentile)
            
        Returns:
            Seuil de probabilité pour les alertes de retournement
        """
        # Prétraiter les données
        X, _ = self.preprocess_data(data, feature_engineering, is_training=False)
        
        # Obtenir les prédictions de retournement
        reversal_probs, _ = self.reversal_detector.predict(X)
        
        # Calculer le seuil basé sur le percentile spécifié
        threshold = np.percentile(reversal_probs, percentile)
        
        return float(threshold)

def test_model():
    """Test simple du modèle pour vérifier qu'il compile correctement"""
    model = EnhancedLSTMModel(
        input_length=60,
        feature_dim=66,
        lstm_units=[64, 48, 32],
        prediction_horizons=[
            (12, "3h", True),
            (48, "12h", True),
            (192, "48h", True)
        ]
    )
    
    # Afficher le résumé du modèle
    model.model.summary()
    
    print("Le modèle a été compilé avec succès !")
    
    return model

if __name__ == "__main__":
    test_model()


