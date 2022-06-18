import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

class TransformerBlock(layers.Layer):
    def __init__(self, embed_dim, num_heads, ff_dim, use_bias=True):
        super(TransformerBlock, self).__init__()
        self.att = layers.MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim, use_bias=use_bias)
        self.ffn = keras.Sequential([
            layers.Dense(ff_dim, activation="relu", use_bias=use_bias), 
            layers.Dense(embed_dim, use_bias=use_bias),
        ])
        self.layernorm1 = layers.LayerNormalization(epsilon=1e-6, center=False, scale=False)
        self.layernorm2 = layers.LayerNormalization(epsilon=1e-6, center=False, scale=False)

    def call(self, inputs, training):
        attn_output = self.att(inputs, inputs)
        out1 = self.layernorm1(inputs + attn_output)
        ffn_output = self.ffn(out1)
        return self.layernorm2(out1 + ffn_output)

class TokenAndPositionEmbedding(layers.Layer):
    def __init__(self, maxlen, vocab_size, embed_dim):
        super(TokenAndPositionEmbedding, self).__init__()
        self.token_emb = layers.Embedding(input_dim=vocab_size, output_dim=embed_dim)
        self.pos_emb = layers.Embedding(input_dim=maxlen, output_dim=embed_dim)

    def call(self, x):
        maxlen = tf.shape(x)[-1]
        positions = tf.range(start=0, limit=maxlen, delta=1)
        positions = self.pos_emb(positions)
        x = self.token_emb(x)
        return x + positions

class TransformerEncoder(layers.Layer):
    def __init__(self, vocab_size: int, maxlen: int, embed_dim: int, ff_dim: int, num_heads: int = 1):
        super(TransformerEncoder, self).__init__()
        self.layers = tf.keras.models.Sequential([
            layers.Input(shape=(maxlen,)),
            TokenAndPositionEmbedding(maxlen, vocab_size, embed_dim),
            TransformerBlock(embed_dim, num_heads, ff_dim),
            layers.GlobalAveragePooling1D(),
            layers.Flatten(),
        ])
    def call(self, x):
        return self.layers(x)
