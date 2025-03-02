"""
Implémentation de mécanismes d'attention avancés pour le modèle LSTM
Inspiré des architectures Transformer avec attention multi-tête
"""
import tensorflow as tf
from tensorflow.keras.layers import Layer, Dense, Reshape, Permute, Concatenate, TimeDistributed, Activation
from tensorflow.keras import backend as K
import numpy as np

class SelfAttention(Layer):
    """
    Mécanisme d'attention qui permet au modèle de se concentrer sur certaines parties d'une séquence
    """
    def __init__(self, attention_units=128, return_attention=False, **kwargs):
        """
        Initialise la couche d'attention
        
        Args:
            attention_units: Nombre d'unités dans la couche d'attention
            return_attention: Si True, retourne également les poids d'attention
        """
        self.attention_units = attention_units
        self.return_attention = return_attention
        super(SelfAttention, self).__init__(**kwargs)
    
    def build(self, input_shape):
        """
        Construit les couches d'attention
        
        Args:
            input_shape: Forme de l'entrée
        """
        # Extraction des dimensions d'entrée
        self.time_steps = input_shape[1]
        self.input_dim = input_shape[2]
        
        # Initialisation des poids pour l'attention
        self.W1 = self.add_weight(name='W1',
                                  shape=(self.input_dim, self.attention_units),
                                  initializer='glorot_uniform',
                                  trainable=True)
        
        self.W2 = self.add_weight(name='W2',
                                  shape=(self.attention_units, 1),
                                  initializer='glorot_uniform',
                                  trainable=True)
        
        super(SelfAttention, self).build(input_shape)
    
    def call(self, inputs, mask=None):
        """
        Applique le mécanisme d'attention
        
        Args:
            inputs: Entrée de forme (batch_size, time_steps, input_dim)
            mask: Masque optionnel
            
        Returns:
            Contexte pondéré par l'attention et poids d'attention si return_attention=True
        """
        # Calcul du score d'attention pour chaque pas de temps
        # et = tanh(W1 * ht)
        et = K.tanh(K.dot(inputs, self.W1))
        
        # at = softmax(W2 * et)
        at = K.dot(et, self.W2)
        at = K.squeeze(at, axis=-1)
        
        # Application du masque si nécessaire
        if mask is not None:
            at *= K.cast(mask, K.floatx())
        
        # Normalisation par softmax
        at = K.softmax(at)
        
        # Calcul du contexte pondéré par l'attention
        # context = at * inputs
        context = K.batch_dot(at, inputs)
        
        if self.return_attention:
            return [context, at]
        return context
    
    def compute_output_shape(self, input_shape):
        """
        Calcule la forme de la sortie
        
        Args:
            input_shape: Forme de l'entrée
            
        Returns:
            Forme de la sortie
        """
        if self.return_attention:
            return [(input_shape[0], self.input_dim), (input_shape[0], self.time_steps)]
        return (input_shape[0], self.input_dim)


class MultiHeadAttention(Layer):
    """
    Attention multi-tête pour capturer différents aspects des séquences temporelles
    Inspiré des architectures Transformer
    """
    def __init__(self, num_heads=4, head_dim=32, dropout=0.1, use_bias=True, return_attention=False, **kwargs):
        """
        Initialise la couche d'attention multi-tête
        
        Args:
            num_heads: Nombre de têtes d'attention
            head_dim: Dimension de chaque tête
            dropout: Taux de dropout
            use_bias: Utiliser un terme de biais
            return_attention: Si True, retourne également les poids d'attention
        """
        super(MultiHeadAttention, self).__init__(**kwargs)
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.dropout = dropout
        self.use_bias = use_bias
        self.return_attention = return_attention
    
    def build(self, input_shape):
        """
        Construit les couches de l'attention multi-tête
        
        Args:
            input_shape: Forme de l'entrée (batch_size, time_steps, input_dim)
        """
        if isinstance(input_shape, list):
            # Si l'entrée est [query, key, value]
            q_shape, k_shape, v_shape = input_shape
            self.query_dim = q_shape[-1]
            self.key_dim = k_shape[-1]
            self.value_dim = v_shape[-1]
        else:
            # Si une seule entrée (self-attention)
            self.query_dim = input_shape[-1]
            self.key_dim = input_shape[-1]
            self.value_dim = input_shape[-1]
        
        self.output_dim = self.num_heads * self.head_dim
        
        # Matrices de projection pour query, key, value
        self.query_weights = self.add_weight(
            name='query_weights',
            shape=(self.query_dim, self.num_heads * self.head_dim),
            initializer='glorot_uniform',
            trainable=True
        )
        
        if self.use_bias:
            self.query_bias = self.add_weight(
                name='query_bias',
                shape=(self.num_heads * self.head_dim,),
                initializer='zeros',
                trainable=True
            )
        
        self.key_weights = self.add_weight(
            name='key_weights',
            shape=(self.key_dim, self.num_heads * self.head_dim),
            initializer='glorot_uniform',
            trainable=True
        )
        
        if self.use_bias:
            self.key_bias = self.add_weight(
                name='key_bias',
                shape=(self.num_heads * self.head_dim,),
                initializer='zeros',
                trainable=True
            )
        
        self.value_weights = self.add_weight(
            name='value_weights',
            shape=(self.value_dim, self.num_heads * self.head_dim),
            initializer='glorot_uniform',
            trainable=True
        )
        
        if self.use_bias:
            self.value_bias = self.add_weight(
                name='value_bias',
                shape=(self.num_heads * self.head_dim,),
                initializer='zeros',
                trainable=True
            )
        
        # Matrice de sortie pour combiner les têtes
        self.output_weights = self.add_weight(
            name='output_weights',
            shape=(self.output_dim, self.value_dim),
            initializer='glorot_uniform',
            trainable=True
        )
        
        if self.use_bias:
            self.output_bias = self.add_weight(
                name='output_bias',
                shape=(self.value_dim,),
                initializer='zeros',
                trainable=True
            )
        
        super(MultiHeadAttention, self).build(input_shape)
    
    def _split_heads(self, x, batch_size):
        """
        Divise la dernière dimension en (num_heads, head_dim)
        
        Args:
            x: Entrée de forme (batch_size, seq_len, num_heads * head_dim)
            batch_size: Taille du batch
            
        Returns:
            Sortie de forme (batch_size, num_heads, seq_len, head_dim)
        """
        x = tf.reshape(x, (batch_size, -1, self.num_heads, self.head_dim))
        return tf.transpose(x, perm=[0, 2, 1, 3])  # (batch_size, num_heads, seq_len, head_dim)
    
    def _combine_heads(self, x, batch_size):
        """
        Combine les têtes pour former (batch_size, seq_len, num_heads * head_dim)
        
        Args:
            x: Entrée de forme (batch_size, num_heads, seq_len, head_dim)
            batch_size: Taille du batch
            
        Returns:
            Sortie de forme (batch_size, seq_len, num_heads * head_dim)
        """
        x = tf.transpose(x, perm=[0, 2, 1, 3])  # (batch_size, seq_len, num_heads, head_dim)
        return tf.reshape(x, (batch_size, -1, self.num_heads * self.head_dim))
    
    def scaled_dot_product_attention(self, q, k, v, mask=None):
        """
        Calcule l'attention avec produit scalaire mis à l'échelle
        
        Args:
            q: Query (batch_size, num_heads, seq_len_q, head_dim)
            k: Key (batch_size, num_heads, seq_len_k, head_dim)
            v: Value (batch_size, num_heads, seq_len_v, head_dim)
            mask: Masque optionnel
            
        Returns:
            Contexte et poids d'attention
        """
        # Produit scalaire entre query et key
        matmul_qk = tf.matmul(q, k, transpose_b=True)  # (batch_size, num_heads, seq_len_q, seq_len_k)
        
        # Mise à l'échelle
        dk = tf.cast(self.head_dim, tf.float32)
        scaled_attention_logits = matmul_qk / tf.math.sqrt(dk)
        
        # Application du masque si fourni
        if mask is not None:
            scaled_attention_logits += (mask * -1e9)
        
        # Softmax sur la dernière dimension (seq_len_k)
        attention_weights = tf.nn.softmax(scaled_attention_logits, axis=-1)
        
        # Application du dropout
        attention_weights = tf.keras.layers.Dropout(self.dropout)(attention_weights)
        
        # Produit avec les valeurs
        output = tf.matmul(attention_weights, v)  # (batch_size, num_heads, seq_len_q, head_dim)
        
        return output, attention_weights
    
    def call(self, inputs, mask=None, training=None):
        """
        Applique l'attention multi-tête
        
        Args:
            inputs: Entrée ou liste [query, key, value]
            mask: Masque optionnel
            training: Indique si c'est l'entraînement
            
        Returns:
            Sortie avec attention et poids d'attention si return_attention=True
        """
        # Gestion de différents types d'entrées
        if isinstance(inputs, list):
            query, key, value = inputs
        else:
            query = key = value = inputs
        
        batch_size = tf.shape(query)[0]
        
        # Projections linéaires et division en têtes
        if self.use_bias:
            query_proj = tf.matmul(query, self.query_weights) + self.query_bias
            key_proj = tf.matmul(key, self.key_weights) + self.key_bias
            value_proj = tf.matmul(value, self.value_weights) + self.value_bias
        else:
            query_proj = tf.matmul(query, self.query_weights)
            key_proj = tf.matmul(key, self.key_weights)
            value_proj = tf.matmul(value, self.value_weights)
        
        # Division en têtes
        query_heads = self._split_heads(query_proj, batch_size)  # (batch_size, num_heads, seq_len_q, head_dim)
        key_heads = self._split_heads(key_proj, batch_size)      # (batch_size, num_heads, seq_len_k, head_dim)
        value_heads = self._split_heads(value_proj, batch_size)  # (batch_size, num_heads, seq_len_v, head_dim)
        
        # Attention avec produit scalaire mis à l'échelle
        attention_output, attention_weights = self.scaled_dot_product_attention(
            query_heads, key_heads, value_heads, mask)
        
        # Combinaison des têtes
        attention_output = self._combine_heads(attention_output, batch_size)  # (batch_size, seq_len_q, output_dim)
        
        # Projection finale
        if self.use_bias:
            output = tf.matmul(attention_output, self.output_weights) + self.output_bias
        else:
            output = tf.matmul(attention_output, self.output_weights)
        
        if self.return_attention:
            return [output, attention_weights]
        return output
    
    def compute_output_shape(self, input_shape):
        """
        Calcule la forme de sortie
        
        Args:
            input_shape: Forme de l'entrée
            
        Returns:
            Forme de la sortie
        """
        if isinstance(input_shape, list):
            q_shape = input_shape[0]
            v_shape = input_shape[2]
            output_shape = (q_shape[0], q_shape[1], v_shape[2])
        else:
            output_shape = (input_shape[0], input_shape[1], input_shape[2])
        
        if self.return_attention:
            if isinstance(input_shape, list):
                q_shape = input_shape[0]
                k_shape = input_shape[1]
                attention_shape = (q_shape[0], self.num_heads, q_shape[1], k_shape[1])
            else:
                attention_shape = (input_shape[0], self.num_heads, input_shape[1], input_shape[1])
            return [output_shape, attention_shape]
        
        return output_shape


class TemporalAttentionBlock(Layer):
    """
    Bloc d'attention temporelle pour séries financières
    Combine l'attention multi-tête avec une connexion résiduelle et normalisation
    """
    def __init__(self, num_heads=4, head_dim=32, ff_dim=128, dropout=0.1, **kwargs):
        """
        Initialise le bloc d'attention temporelle
        
        Args:
            num_heads: Nombre de têtes d'attention
            head_dim: Dimension de chaque tête
            ff_dim: Dimension du feed-forward
            dropout: Taux de dropout
        """
        super(TemporalAttentionBlock, self).__init__(**kwargs)
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.ff_dim = ff_dim
        self.dropout = dropout
    
    def build(self, input_shape):
        """
        Construit les couches du bloc d'attention
        
        Args:
            input_shape: Forme de l'entrée
        """
        self.attention = MultiHeadAttention(
            num_heads=self.num_heads,
            head_dim=self.head_dim,
            dropout=self.dropout
        )
        
        self.ff1 = Dense(self.ff_dim, activation='relu')
        self.ff2 = Dense(input_shape[-1])
        
        self.dropout1 = tf.keras.layers.Dropout(self.dropout)
        self.dropout2 = tf.keras.layers.Dropout(self.dropout)
        
        self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        
        super(TemporalAttentionBlock, self).build(input_shape)
    
    def call(self, inputs, training=None, mask=None):
        """
        Applique le bloc d'attention
        
        Args:
            inputs: Entrée de forme (batch_size, seq_len, features)
            training: Indique si c'est l'entraînement
            mask: Masque optionnel
            
        Returns:
            Sortie du bloc d'attention
        """
        # Sous-couche d'attention multi-tête
        attn_output = self.attention(inputs, mask=mask)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(inputs + attn_output)  # Connexion résiduelle
        
        # Sous-couche feed-forward
        ff_output = self.ff1(out1)
        ff_output = self.ff2(ff_output)
        ff_output = self.dropout2(ff_output, training=training)
        
        # Connexion résiduelle finale
        return self.layernorm2(out1 + ff_output)


class TimeSeriesAttention(Layer):
    """
    Mécanisme d'attention spécialisé pour séries temporelles financières
    """
    def __init__(self, filters=64, kernel_size=1, **kwargs):
        """
        Initialise la couche d'attention
        
        Args:
            filters: Nombre de filtres convolutifs
            kernel_size: Taille du noyau convolutif
        """
        super(TimeSeriesAttention, self).__init__(**kwargs)
        self.filters = filters
        self.kernel_size = kernel_size
    
    def build(self, input_shape):
        """
        Construit les couches de l'attention
        
        Args:
            input_shape: Forme de l'entrée
        """
        # Extraction des dimensions
        self.time_steps = input_shape[1]
        self.input_dim = input_shape[2]
        
        # Couche de réduction de dimension temporelle
        self.conv_qkv = tf.keras.layers.Conv1D(
            filters=self.filters * 3,  # Pour query, key et value
            kernel_size=self.kernel_size,
            padding='same',
            use_bias=True
        )
        
        # Couche de sortie
        self.conv_out = tf.keras.layers.Conv1D(
            filters=self.input_dim,
            kernel_size=self.kernel_size,
            padding='same',
            use_bias=True
        )
        
        super(TimeSeriesAttention, self).build(input_shape)
    
    def call(self, inputs, mask=None):
        """
        Applique le mécanisme d'attention
        
        Args:
            inputs: Entrée de forme (batch_size, time_steps, input_dim)
            mask: Masque optionnel
            
        Returns:
            Sortie avec attention
        """
        # Projection QKV
        qkv = self.conv_qkv(inputs)
        
        # Séparation en query, key, value
        batch_size = tf.shape(qkv)[0]
        q, k, v = tf.split(qkv, 3, axis=-1)
        
        # Calcul du score d'attention
        # Produit scalaire de q et k, puis mise à l'échelle
        score = tf.matmul(q, k, transpose_b=True)
        scale = tf.sqrt(tf.cast(self.filters, tf.float32))
        score = score / scale
        
        # Application du masque si nécessaire
        if mask is not None:
            score += (1.0 - mask) * -1e9
        
        # Appliquer softmax pour obtenir les poids d'attention
        attention_weights = tf.nn.softmax(score, axis=-1)
        
        # Appliquer l'attention aux valeurs
        context = tf.matmul(attention_weights, v)
        
        # Projection finale
        output = self.conv_out(context)
        
        # Connexion résiduelle
        return inputs + output