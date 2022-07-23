import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

class FNetBlock(layers.Layer):
    def __init__(self, embed_dim, ff_dim, use_bias=True):
        super(FNetBlock, self).__init__()
        self.ffn = keras.Sequential([
            layers.Dense(ff_dim, activation="gelu", use_bias=use_bias),
            layers.Dense(embed_dim, use_bias=use_bias),
        ])
        self.layernorm1 = layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = layers.LayerNormalization(epsilon=1e-6)

    def call(self, inputs):
        x_complex = tf.cast(inputs, tf.complex64)
        attn_output = tf.cast(tf.math.real(tf.signal.fft2d(x_complex)), tf.float32)
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

class FNet(layers.Layer):
    def __init__(self, vocab_size: int, maxlen: int, embed_dim: int, num_blocks: int):
        super(FNet, self).__init__()
        self.layers = tf.keras.models.Sequential([
            layers.Input(shape=(maxlen,)),
            TokenAndPositionEmbedding(maxlen, vocab_size, embed_dim),
        ])
        for _ in range(num_blocks):
            self.layers.add(FNetBlock(embed_dim, embed_dim))

    def call(self, x):
        return self.layers(x)
