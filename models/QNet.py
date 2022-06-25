import tensorflow as tf
from tensorflow.keras import layers
import tensorflow_quantum as tfq

import cirq
import sympy
import numpy as np
from numpy import pi

def quantum_data_encoder(bits, symbols):
    circuit = cirq.Circuit()
    for bit in bits: circuit.append(cirq.H(bit))
    for idx, bit in enumerate(bits):
        circuit.append(cirq.Z(bit)**symbols[idx])
    return circuit

def quanttention(bits, embed_size):
    circuit = cirq.Circuit()
    seq_len = len(bits) // embed_size
    for i in range(embed_size):
        subs = [ bits[j * embed_size + i] for j in range(seq_len) ]

        for j in range(seq_len-1, -1, -1):
            circuit.append(cirq.H(subs[j]))
            for k in range(0, j):
                circuit.append(cirq.CZ(subs[k], subs[j])**(pi/2**(j-k)))
        for j in range(seq_len//2):
            circuit.append(cirq.SWAP(subs[j], subs[seq_len-j-1]))
    return circuit

def quantum_feedforward(bits, embed_size, level, symbols):
    circuit = cirq.Circuit()
    for idx, bit in enumerate(bits):
        circuit.append(cirq.X(bit) ** symbols[(idx + level) % embed_size])
        circuit.append(cirq.Y(bit) ** symbols[(idx + level) % embed_size + embed_size])
        circuit.append(cirq.Z(bit) ** symbols[(idx + level) % embed_size + 2 * embed_size])
    for i in range(len(bits)):
        circuit.append(cirq.CX(bits[(i + level) % len(bits)], bits[(i + level + 1) % len(bits)]))
    return circuit

def generate_model(embed_size, seq_len, depth = 1):
    qubits = cirq.GridQubit.rect(seq_len, embed_size)
    circuit = cirq.Circuit()
    
    embedding_symbols = sympy.symbols(f'e0:{embed_size * seq_len}')
    circuit = quantum_data_encoder(qubits, embedding_symbols)
    
    ff_symbols = sympy.symbols(f't0:{embed_size * depth * 3}')
    for d in range(depth):
        circuit += quanttention(qubits, embed_size)
        symbols = ff_symbols[d * embed_size * 3 : (d+1) * embed_size * 3]
        circuit += quantum_feedforward(qubits, embed_size, d, symbols)
    return qubits, circuit


class ParametersLayer(layers.Layer):
    def __init__(self, vocab_size, embed_dim, depth):
        super(ParametersLayer, self).__init__()
        self.embedding = layers.Embedding(input_dim=vocab_size, output_dim=embed_dim)
        self.parameters = tf.Variable(
            np.random.uniform(0, 2 * np.pi, (1, depth * 3 * embed_dim)),
            name="Q_param",
            dtype=tf.float32,
        )
        self.flatten = layers.Flatten()

    def call(self, inputs):
        embedding = self.embedding(inputs)
        x1 = self.flatten(embedding)
        x2 = tf.tile(self.parameters, tf.stack([tf.shape(inputs)[0], 1]))
        x = tf.concat([x1, x2], -1)
        return x

class QNet(layers.Layer):
    def __init__(self, embed_dim, seq_len, depth):
        super(QNet, self).__init__()
        qubits, model_circuit = generate_model(embed_dim, seq_len, depth)
        observables = [ cirq.Z(bit) for bit in qubits ]
        self.backbone = tfq.layers.ControlledPQC(model_circuit, operators=observables)
    
    def call(self, inputs):
        empty_circuit = tf.tile(tfq.convert_to_tensor([cirq.Circuit()]), tf.stack([tf.shape(inputs)[0]]))
        y = self.backbone([empty_circuit, inputs])
        
        return y

class QNetEncoder(layers.Layer):
    def __init__(self, vocab_size: int, maxlen: int, embed_dim: int, ff_dim: int, num_heads: int = 1):
        super(QNetEncoder, self).__init__()
        self.layers = tf.keras.models.Sequential([
            layers.Input(shape=(maxlen,)),
            ParametersLayer(vocab_size, embed_dim, 1),
            QNet(embed_dim, maxlen, 1),
        ])
    def call(self, x):
        return self.layers(x)
