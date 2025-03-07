"""
Attention mechanisms for deep learning models
Includes implementations of various attention mechanisms for time series analysis
"""
import tensorflow as tf
from tensorflow.keras.layers import Layer
import numpy as np

class TimeSeriesAttention(Layer):
    """
    Simple time series attention mechanism that focuses on important time steps
    """
    def __init__(self, filters=32, kernel_size=1, **kwargs):
        super(TimeSeriesAttention, self).__init__(**kwargs)
        self.filters = filters
        self.kernel_size = kernel_size
        
    def build(self, input_shape):
        # Shape of input should be (batch_size, time_steps, features)
        self.conv1d = tf.keras.layers.Conv1D(
            filters=self.filters,
            kernel_size=self.kernel_size,
            padding="same",
            activation="tanh"
        )
        self.dense = tf.keras.layers.Dense(1)
        super(TimeSeriesAttention, self).build(input_shape)
    
    def call(self, inputs):
        # Calculate attention weights
        x = self.conv1d(inputs)
        e = self.dense(x)  # (batch_size, time_steps, 1)
        
        # Convert attention weights to probabilities (softmax over time dimension)
        alpha = tf.nn.softmax(e, axis=1)  # (batch_size, time_steps, 1)
        
        # Apply attention weights to input sequence
        attended = inputs * alpha
        
        return attended
    
    def compute_output_shape(self, input_shape):
        return input_shape
        
    def get_config(self):
        config = super(TimeSeriesAttention, self).get_config()
        config.update({
            "filters": self.filters,
            "kernel_size": self.kernel_size
        })
        return config


class TemporalAttentionBlock(Layer):
    """
    Temporal attention block for focusing on relevant time steps in a sequence
    Adapted for financial time series analysis
    """
    def __init__(self, units=32, return_attention=False, **kwargs):
        super(TemporalAttentionBlock, self).__init__(**kwargs)
        self.units = units
        self.return_attention = return_attention
        
    def build(self, input_shape):
        # Shape of input should be (batch_size, time_steps, features)
        self.W = self.add_weight(
            name="attention_weight",
            shape=(input_shape[-1], self.units),
            initializer="glorot_uniform",
            trainable=True
        )
        self.b = self.add_weight(
            name="attention_bias",
            shape=(self.units,),
            initializer="zeros",
            trainable=True
        )
        self.u = self.add_weight(
            name="context_vector",
            shape=(self.units, 1),
            initializer="glorot_uniform",
            trainable=True
        )
        super(TemporalAttentionBlock, self).build(input_shape)
    
    def call(self, inputs):
        # Shape: (batch_size, time_steps, features) @ (features, units) + (units,)
        # Result: (batch_size, time_steps, units)
        uit = tf.tanh(tf.tensordot(inputs, self.W, axes=1) + self.b)
        
        # Shape: (batch_size, time_steps, units) @ (units, 1)
        # Result: (batch_size, time_steps, 1)
        scores = tf.tensordot(uit, self.u, axes=1)
        
        # Apply softmax to get attention weights (over time dimension)
        # Shape: (batch_size, time_steps, 1)
        attention_weights = tf.nn.softmax(scores, axis=1)
        
        # Apply attention weights to the input sequence
        # Shape: (batch_size, time_steps, features) * (batch_size, time_steps, 1)
        # Result: (batch_size, time_steps, features)
        weighted_inputs = inputs * attention_weights
        
        # Sum over time dimension to get context vector
        # Shape: (batch_size, features)
        context = tf.reduce_sum(weighted_inputs, axis=1)
        
        if self.return_attention:
            return [context, attention_weights]
        
        return context
    
    def compute_output_shape(self, input_shape):
        if self.return_attention:
            return [(input_shape[0], input_shape[2]), (input_shape[0], input_shape[1], 1)]
        return (input_shape[0], input_shape[2])
        
    def get_config(self):
        config = super(TemporalAttentionBlock, self).get_config()
        config.update({
            "units": self.units,
            "return_attention": self.return_attention
        })
        return config


class MultiHeadAttention(Layer):
    """
    Multi-head attention mechanism inspired by the Transformer architecture
    Adapted for time series analysis
    """
    def __init__(self, num_heads=8, head_dim=32, dropout=0.1, use_residual=True, **kwargs):
        super(MultiHeadAttention, self).__init__(**kwargs)
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.output_dim = num_heads * head_dim
        self.dropout_rate = dropout
        self.use_residual = use_residual
        
    def build(self, input_shape):
        self.query_dense = tf.keras.layers.Dense(self.output_dim, use_bias=False)
        self.key_dense = tf.keras.layers.Dense(self.output_dim, use_bias=False)
        self.value_dense = tf.keras.layers.Dense(self.output_dim, use_bias=False)
        
        self.output_dense = tf.keras.layers.Dense(input_shape[-1])
        self.dropout = tf.keras.layers.Dropout(self.dropout_rate)
        
        if self.use_residual:
            self.layer_norm = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        
        super(MultiHeadAttention, self).build(input_shape)
    
    def call(self, inputs, training=None):
        # Original inputs shape: (batch_size, time_steps, features)
        
        # Linear projections
        query = self.query_dense(inputs)  # (batch_size, time_steps, output_dim)
        key = self.key_dense(inputs)      # (batch_size, time_steps, output_dim)
        value = self.value_dense(inputs)  # (batch_size, time_steps, output_dim)
        
        # Reshape for multi-head attention
        batch_size = tf.shape(query)[0]
        time_steps = tf.shape(query)[1]
        
        # Reshape to (batch_size, time_steps, num_heads, head_dim)
        query = tf.reshape(query, [batch_size, time_steps, self.num_heads, self.head_dim])
        key = tf.reshape(key, [batch_size, time_steps, self.num_heads, self.head_dim])
        value = tf.reshape(value, [batch_size, time_steps, self.num_heads, self.head_dim])
        
        # Transpose to (batch_size, num_heads, time_steps, head_dim)
        query = tf.transpose(query, [0, 2, 1, 3])
        key = tf.transpose(key, [0, 2, 1, 3])
        value = tf.transpose(value, [0, 2, 1, 3])
        
        # Scaled dot-product attention
        # matmul_qk shape: (batch_size, num_heads, time_steps, time_steps)
        matmul_qk = tf.matmul(query, key, transpose_b=True)
        
        # Scale matmul_qk
        dk = tf.cast(self.head_dim, tf.float32)
        scaled_attention_logits = matmul_qk / tf.math.sqrt(dk)
        
        # Apply softmax to get attention weights
        attention_weights = tf.nn.softmax(scaled_attention_logits, axis=-1)
        
        # Apply dropout to attention weights
        attention_weights = self.dropout(attention_weights, training=training)
        
        # Apply attention weights to value
        # output shape: (batch_size, num_heads, time_steps, head_dim)
        output = tf.matmul(attention_weights, value)
        
        # Reshape back to original shape
        # First transpose back to (batch_size, time_steps, num_heads, head_dim)
        output = tf.transpose(output, [0, 2, 1, 3])
        
        # Then reshape to (batch_size, time_steps, output_dim)
        output = tf.reshape(output, [batch_size, time_steps, self.output_dim])
        
        # Final linear projection
        output = self.output_dense(output)
        
        # Apply residual connection and layer normalization if specified
        if self.use_residual:
            output = self.layer_norm(output + inputs)
        
        return output
    
    def compute_output_shape(self, input_shape):
        return input_shape
        
    def get_config(self):
        config = super(MultiHeadAttention, self).get_config()
        config.update({
            "num_heads": self.num_heads,
            "head_dim": self.head_dim,
            "dropout_rate": self.dropout_rate,
            "use_residual": self.use_residual
        })
        return config


class CrossTimeAttention(Layer):
    """
    Cross-time attention mechanism that enables looking at relationships between
    different time scales (e.g., short-term patterns vs long-term trends)
    """
    def __init__(self, num_time_scales=3, units=32, **kwargs):
        super(CrossTimeAttention, self).__init__(**kwargs)
        self.num_time_scales = num_time_scales
        self.units = units
        
    def build(self, input_shape):
        # Input shape should be a list of tensors, each with shape (batch, time_steps, features)
        if not isinstance(input_shape, list) or len(input_shape) != self.num_time_scales:
            raise ValueError(f"Input should be a list of {self.num_time_scales} tensors")
        
        # Create projection layers for each time scale
        self.projections = []
        for i in range(self.num_time_scales):
            self.projections.append(tf.keras.layers.Dense(self.units))
        
        # Create output projection
        self.output_projection = tf.keras.layers.Dense(input_shape[0][-1])
        
        super(CrossTimeAttention, self).build(input_shape)
    
    def call(self, inputs):
        # Inputs should be a list of tensors for different time scales
        if not isinstance(inputs, list) or len(inputs) != self.num_time_scales:
            raise ValueError(f"Expected a list of {self.num_time_scales} tensors, got {len(inputs)}")
        
        # Project each input to the same dimension
        projected = [proj(inp) for proj, inp in zip(self.projections, inputs)]
        
        # Calculate attention scores between each pair of time scales
        attention_maps = []
        for i in range(self.num_time_scales):
            for j in range(self.num_time_scales):
                if i != j:
                    # Calculate attention from time scale i to time scale j
                    # Shape: (batch, time_i, time_j)
                    attention = tf.matmul(projected[i], projected[j], transpose_b=True)
                    
                    # Scale and apply softmax
                    scale = tf.math.sqrt(tf.cast(self.units, tf.float32))
                    attention = tf.nn.softmax(attention / scale, axis=-1)
                    
                    # Apply attention to get context
                    # Shape: (batch, time_i, features_j)
                    context = tf.matmul(attention, inputs[j])
                    
                    attention_maps.append(context)
        
        # Combine all contexts (simple mean for now)
        combined = tf.reduce_mean(tf.stack(attention_maps, axis=0), axis=0)
        
        # Project back to original feature dimension
        output = self.output_projection(combined)
        
        # Add residual connection with the primary input (first in the list)
        output = output + inputs[0]
        
        return output
    
    def compute_output_shape(self, input_shape):
        return input_shape[0]
        
    def get_config(self):
        config = super(CrossTimeAttention, self).get_config()
        config.update({
            "num_time_scales": self.num_time_scales,
            "units": self.units
        })
        return config


class FeatureAttention(Layer):
    """
    Cross-feature attention module to capture relationships between different features
    Useful for extracting which features are most important at different times
    """
    def __init__(self, use_residual=True, **kwargs):
        super(FeatureAttention, self).__init__(**kwargs)
        self.use_residual = use_residual
        
    def build(self, input_shape):
        # Shape of input should be (batch_size, time_steps, features)
        self.feature_dense = tf.keras.layers.Dense(input_shape[-1], use_bias=False)
        self.time_dense = tf.keras.layers.Dense(input_shape[1], use_bias=False)
        
        if self.use_residual:
            self.layer_norm = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        
        super(FeatureAttention, self).build(input_shape)
    
    def call(self, inputs):
        # Calculate feature attention
        # First, transpose to (batch_size, features, time_steps)
        transposed = tf.transpose(inputs, [0, 2, 1])
        
        # Calculate attention scores between features
        # Shape: (batch_size, features, features)
        feature_attention = tf.matmul(
            self.feature_dense(transposed), 
            transposed, 
            transpose_b=True
        )
        
        # Apply softmax to get feature attention weights
        feature_attention = tf.nn.softmax(feature_attention, axis=-1)
        
        # Apply feature attention
        # Shape: (batch_size, features, time_steps)
        feature_context = tf.matmul(feature_attention, transposed)
        
        # Transpose back to (batch_size, time_steps, features)
        feature_context = tf.transpose(feature_context, [0, 2, 1])
        
        # Calculate time attention
        # Shape: (batch_size, time_steps, time_steps)
        time_attention = tf.matmul(
            self.time_dense(inputs),
            inputs,
            transpose_b=True
        )
        
        # Apply softmax to get time attention weights
        time_attention = tf.nn.softmax(time_attention, axis=-1)
        
        # Apply time attention
        # Shape: (batch_size, time_steps, features)
        time_context = tf.matmul(time_attention, inputs)
        
        # Combine feature and time context (simple average)
        combined = (feature_context + time_context) / 2.0
        
        # Apply residual connection if specified
        if self.use_residual:
            combined = self.layer_norm(combined + inputs)
        
        return combined
    
    def compute_output_shape(self, input_shape):
        return input_shape
        
    def get_config(self):
        config = super(FeatureAttention, self).get_config()
        config.update({
            "use_residual": self.use_residual
        })
        return config


# Testing utilities
def test_attention_mechanism():
    """Test the attention mechanisms on random data"""
    # Create random input
    batch_size = 32
    time_steps = 60
    features = 30
    
    inputs = tf.random.normal((batch_size, time_steps, features))
    
    # Test TimeSeriesAttention
    attention1 = TimeSeriesAttention(filters=32)
    output1 = attention1(inputs)
    print(f"TimeSeriesAttention output shape: {output1.shape}")
    
    # Test TemporalAttentionBlock
    attention2 = TemporalAttentionBlock(units=32, return_attention=True)
    output2 = attention2(inputs)
    print(f"TemporalAttentionBlock output shapes: {[out.shape for out in output2]}")
    
    # Test MultiHeadAttention
    attention3 = MultiHeadAttention(num_heads=8, head_dim=16)
    output3 = attention3(inputs)
    print(f"MultiHeadAttention output shape: {output3.shape}")
    
    # Test FeatureAttention
    attention4 = FeatureAttention()
    output4 = attention4(inputs)
    print(f"FeatureAttention output shape: {output4.shape}")
    
    print("All attention mechanisms tested successfully!")

if __name__ == "__main__":
    test_attention_mechanism()