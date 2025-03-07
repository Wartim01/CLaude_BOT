"""
Transformer model for time series forecasting in cryptocurrency markets
Based on the Transformer architecture with adaptations for financial time series
"""
import os
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input, Dense, Dropout, LayerNormalization, Activation,
    Conv1D, GlobalAveragePooling1D, Flatten, Concatenate, Add, Lambda
)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
import tensorflow.keras.backend as K
from typing import Dict, List, Tuple, Optional
import json
from datetime import datetime

from config.config import DATA_DIR
from utils.logger import setup_logger
from ai.models.attention import MultiHeadAttention, TemporalAttentionBlock, CrossTimeAttention

logger = setup_logger("transformer_model")

class PositionalEncoding(tf.keras.layers.Layer):
    """
    Positional encoding layer for Transformer model to provide time step information
    """
    def __init__(self, max_position=2000, d_model=128, **kwargs):
        super(PositionalEncoding, self).__init__(**kwargs)
        self.max_position = max_position
        self.d_model = d_model
        
    def build(self, input_shape):
        # Create positional encoding matrix
        position = np.arange(self.max_position)[:, np.newaxis]
        div_term = np.exp(np.arange(0, self.d_model, 2) * -(np.log(10000.0) / self.d_model))
        
        pos_encoding = np.zeros((self.max_position, self.d_model))
        pos_encoding[:, 0::2] = np.sin(position * div_term)
        pos_encoding[:, 1::2] = np.cos(position * div_term)
        
        # Create a non-trainable weight for positional encoding
        self.pos_encoding = self.add_weight(
            name='pos_encoding',
            shape=(1, self.max_position, self.d_model),
            initializer=tf.keras.initializers.Constant(pos_encoding),
            trainable=False
        )
        
        super(PositionalEncoding, self).build(input_shape)
    
    def call(self, inputs):
        # Get sequence length
        seq_len = tf.shape(inputs)[1]
        
        # Apply positional encoding to input
        return inputs + self.pos_encoding[:, :seq_len, :]
    
    def compute_output_shape(self, input_shape):
        return input_shape
    
    def get_config(self):
        config = super(PositionalEncoding, self).get_config()
        config.update({
            'max_position': self.max_position,
            'd_model': self.d_model
        })
        return config

def transformer_encoder_block(inputs, head_size, num_heads, ff_dim, dropout=0.1):
    """
    Transformer encoder block with multi-head attention and feed-forward network
    
    Args:
        inputs: Input tensor
        head_size: Size of each attention head
        num_heads: Number of attention heads
        ff_dim: Hidden dimensionality of the feed-forward network
        dropout: Dropout rate
        
    Returns:
        Output tensor after applying transformer encoder block
    """
    # Multi-head attention
    attention = MultiHeadAttention(
        num_heads=num_heads, 
        head_dim=head_size,
        dropout=dropout,
        use_residual=True
    )(inputs)
    
    # Feed-forward network
    x = LayerNormalization(epsilon=1e-6)(attention)
    x = Dense(ff_dim, activation='relu')(x)
    x = Dropout(dropout)(x)
    x = Dense(inputs.shape[-1])(x)
    x = Dropout(dropout)(x)
    
    # Residual connection
    outputs = Add()([x, attention])
    outputs = LayerNormalization(epsilon=1e-6)(outputs)
    
    return outputs

class TransformerModel:
    """
    Transformer model for cryptocurrency price prediction
    
    Implements a transformer-based architecture for multi-horizon prediction
    with various output factors (direction, volatility, etc.)
    """
    def __init__(self, 
                input_length: int = 60,
                feature_dim: int = 30,
                num_layers: int = 4,
                num_heads: int = 8,
                head_size: int = 32,
                ff_dim: int = 128,
                dropout_rate: float = 0.1,
                learning_rate: float = 0.0001,
                prediction_horizons: List[Tuple[int, str, bool]] = [
                    (12, "3h", True),    # 3 hours with 15min candles
                    (48, "12h", True),   # 12 hours
                    (192, "48h", True)   # 48 hours
                ]):
        """
        Initialize the transformer model
        
        Args:
            input_length: Input sequence length (time steps)
            feature_dim: Number of input features
            num_layers: Number of transformer encoder layers
            num_heads: Number of attention heads
            head_size: Size of each attention head
            ff_dim: Feed-forward network hidden dimension
            dropout_rate: Dropout rate
            learning_rate: Learning rate for optimizer
            prediction_horizons: List of horizons to predict (periods, name, is_main)
        """
        self.input_length = input_length
        self.feature_dim = feature_dim
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.head_size = head_size
        self.ff_dim = ff_dim
        self.dropout_rate = dropout_rate
        self.learning_rate = learning_rate
        self.prediction_horizons = prediction_horizons
        
        # Extract just the periods for compatibility
        self.horizon_periods = [h[0] for h in prediction_horizons]
        
        # Output factors
        self.factors = ["direction", "volatility", "volume", "momentum"]
        self.num_factors = len(self.factors)
        
        # Build the model
        self.model = self._build_model()
        
        # Model path
        self.models_dir = os.path.join(DATA_DIR, "models", "production")
        os.makedirs(self.models_dir, exist_ok=True)
        self.model_path = os.path.join(self.models_dir, "transformer_model.keras")
        
        # Training history
        self.training_history = []
        
    def _build_model(self) -> Model:
        """
        Build the transformer model architecture
        
        Returns:
            Compiled Keras model
        """
        # Input layer
        inputs = Input(shape=(self.input_length, self.feature_dim), name="market_sequence")
        
        # Initial projection to match desired model dimension
        d_model = self.head_size * self.num_heads
        x = Dense(d_model, activation='linear')(inputs)
        
        # Add positional encoding
        x = PositionalEncoding(
            max_position=self.input_length,
            d_model=d_model
        )(x)
        
        # Apply dropout
        x = Dropout(self.dropout_rate)(x)
        
        # Stack transformer encoder blocks
        for i in range(self.num_layers):
            x = transformer_encoder_block(
                x, 
                head_size=self.head_size,
                num_heads=self.num_heads,
                ff_dim=self.ff_dim,
                dropout=self.dropout_rate
            )
        
        # Global context
        context = GlobalAveragePooling1D()(x)
        
        # Dense layers for final prediction
        context = Dense(d_model, activation='relu')(context)
        context = LayerNormalization(epsilon=1e-6)(context)
        context = Dropout(self.dropout_rate)(context)
        
        # Create outputs for each horizon and factor
        outputs = []
        output_names = []
        losses = []
        metrics = []
        
        for h_idx, (horizon, horizon_name, _) in enumerate(self.prediction_horizons):
            # Horizon-specific features
            horizon_features = Dense(
                64,
                activation='relu',
                name=f"horizon_{horizon_name}_features"
            )(context)
            
            # Direction (probability of upward movement)
            direction = Dense(
                1, 
                activation='sigmoid',
                name=f"direction_{horizon_name}"
            )(horizon_features)
            outputs.append(direction)
            output_names.append(f"direction_{horizon_name}")
            losses.append('binary_crossentropy')
            metrics.append('accuracy')
            
            # Volatility (relative)
            volatility = Dense(
                1, 
                activation='relu',  # Volatility is always positive
                name=f"volatility_{horizon_name}"
            )(horizon_features)
            outputs.append(volatility)
            output_names.append(f"volatility_{horizon_name}")
            losses.append('mse')
            metrics.append('mae')
            
            # Volume relative
            volume = Dense(
                1, 
                activation='relu',  # Volume ratio is always positive
                name=f"volume_{horizon_name}"
            )(horizon_features)
            outputs.append(volume)
            output_names.append(f"volume_{horizon_name}")
            losses.append('mse')
            metrics.append('mae')
            
            # Momentum (trend strength)
            momentum = Dense(
                1, 
                activation='tanh',  # Tanh for values between -1 and 1
                name=f"momentum_{horizon_name}"
            )(horizon_features)
            outputs.append(momentum)
            output_names.append(f"momentum_{horizon_name}")
            losses.append('mse')
            metrics.append('mae')
        
        # Create and compile the model
        model = Model(inputs=inputs, outputs=outputs, name="transformer_model")
        
        model.compile(
            optimizer=Adam(learning_rate=self.learning_rate),
            loss=losses,
            metrics=metrics
        )
        
        return model
    
    def summary(self):
        """Print model summary"""
        return self.model.summary()
    
    def save(self, path: Optional[str] = None) -> None:
        """
        Save the model to disk
        
        Args:
            path: Path to save the model (uses default if None)
        """
        save_path = path or self.model_path
        
        # Create directory if needed
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        
        # Save the model
        self.model.save(save_path)
        logger.info(f"Model saved: {save_path}")
        
        # Save model configuration
        config_path = os.path.splitext(save_path)[0] + "_config.json"
        with open(config_path, 'w') as f:
            json.dump({
                "input_length": self.input_length,
                "feature_dim": self.feature_dim,
                "num_layers": self.num_layers,
                "num_heads": self.num_heads,
                "head_size": self.head_size,
                "ff_dim": self.ff_dim,
                "dropout_rate": self.dropout_rate,
                "learning_rate": self.learning_rate,
                "prediction_horizons": self.prediction_horizons,
                "saved_at": datetime.now().isoformat()
            }, f, indent=2, default=str)
    
    def load(self, path: Optional[str] = None) -> None:
        """
        Load model from disk
        
        Args:
            path: Path to load the model from (uses default if None)
        """
        load_path = path or self.model_path
        
        if not os.path.exists(load_path):
            raise FileNotFoundError(f"Model not found: {load_path}")
        
        # Load the model
        self.model = tf.keras.models.load_model(
            load_path,
            custom_objects={
                'PositionalEncoding': PositionalEncoding,
                'MultiHeadAttention': MultiHeadAttention,
                'TemporalAttentionBlock': TemporalAttentionBlock,
                'CrossTimeAttention': CrossTimeAttention
            }
        )
        logger.info(f"Model loaded: {load_path}")
        
        # Load model configuration if available
        config_path = os.path.splitext(load_path)[0] + "_config.json"
        if os.path.exists(config_path):
            try:
                with open(config_path, 'r') as f:
                    config = json.load(f)
                    
                # Update model attributes from config
                self.input_length = config.get("input_length", self.input_length)
                self.feature_dim = config.get("feature_dim", self.feature_dim)
                self.num_layers = config.get("num_layers", self.num_layers)
                self.num_heads = config.get("num_heads", self.num_heads)
                self.head_size = config.get("head_size", self.head_size)
                self.ff_dim = config.get("ff_dim", self.ff_dim)
                self.dropout_rate = config.get("dropout_rate", self.dropout_rate)
                self.learning_rate = config.get("learning_rate", self.learning_rate)
                self.prediction_horizons = config.get("prediction_horizons", self.prediction_horizons)
                
                # Update horizon periods
                self.horizon_periods = [h[0] for h in self.prediction_horizons]
                
                logger.info(f"Model configuration loaded from {config_path}")
            except Exception as e:
                logger.warning(f"Error loading model configuration: {str(e)}")

    def train(self, X_train, y_train, X_val=None, y_val=None, 
              epochs=100, batch_size=32, callbacks=None) -> dict:
        """
        Train the transformer model
        
        Args:
            X_train: Training data features
            y_train: Training data targets
            X_val: Validation data features
            y_val: Validation data targets 
            epochs: Number of training epochs
            batch_size: Batch size for training
            callbacks: List of Keras callbacks
            
        Returns:
            Training history
        """
        # Default callbacks if none provided
        if callbacks is None:
            callbacks = [
                EarlyStopping(
                    monitor='val_loss' if X_val is not None else 'loss',
                    patience=15,
                    restore_best_weights=True
                ),
                ReduceLROnPlateau(
                    monitor='val_loss' if X_val is not None else 'loss',
                    factor=0.5,
                    patience=7,
                    min_lr=1e-6,
                    verbose=1
                )
            ]
            
            # Add model checkpoint callback
            checkpoint_dir = os.path.join(DATA_DIR, "models", "checkpoints")
            os.makedirs(checkpoint_dir, exist_ok=True)
            
            checkpoint_path = os.path.join(
                checkpoint_dir, 
                f"transformer_{datetime.now().strftime('%Y%m%d_%H%M')}_best.keras"
            )
            
            callbacks.append(
                ModelCheckpoint(
                    filepath=checkpoint_path,
                    monitor='val_loss' if X_val is not None else 'loss',
                    save_best_only=True,
                    save_weights_only=False,
                    verbose=1
                )
            )
        
        # Prepare validation data
        validation_data = (X_val, y_val) if X_val is not None and y_val is not None else None
        
        # Train the model
        history = self.model.fit(
            x=X_train,
            y=y_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_data=validation_data,
            callbacks=callbacks,
            verbose=1
        )
        
        # Save training history
        self.training_history = history.history
        
        # Return the training history
        return history.history
    
    def predict(self, X):
        """
        Make predictions with the transformer model
        
        Args:
            X: Input features
            
        Returns:
            Predictions for each output
        """
        return self.model.predict(X)
    
    def evaluate(self, X, y):
        """
        Evaluate the model performance
        
        Args:
            X: Input features
            y: Target values
            
        Returns:
            Evaluation metrics
        """
        return self.model.evaluate(X, y, verbose=0)
    
    def predict_market_direction(self, X, horizon_idx=0):
        """
        Predict market direction (up/down) for a specific horizon
        
        Args:
            X: Input features
            horizon_idx: Index of the horizon to predict (0=short, 1=medium, 2=long)
            
        Returns:
            Array of probabilities (>0.5 indicates upward movement)
        """
        predictions = self.model.predict(X)
        
        # Each horizon has multiple outputs (direction, volatility, etc.)
        # For direction, we take the first output for the specified horizon
        direction_idx = horizon_idx * self.num_factors
        
        return predictions[direction_idx]
    
    def get_full_prediction(self, X):
        """
        Get a structured prediction with all outputs
        
        Args:
            X: Input features
            
        Returns:
            Dictionary with structured predictions
        """
        raw_predictions = self.model.predict(X)
        
        # Create a structured output
        result = {}
        
        for h_idx, (horizon, horizon_name, _) in enumerate(self.prediction_horizons):
            # Calculate base index for this horizon
            base_idx = h_idx * self.num_factors
            
            # Extract predictions for each factor
            direction_prob = raw_predictions[base_idx]
            volatility = raw_predictions[base_idx + 1]
            volume = raw_predictions[base_idx + 2]
            momentum = raw_predictions[base_idx + 3]
            
            # Store in result dictionary
            result[horizon_name] = {
                "direction_probability": direction_prob,
                "volatility": volatility,
                "volume_relative": volume,
                "momentum": momentum
            }
        
        return result