import tensorflow as tf
from tensorflow.keras import layers
import tensorflow_quantum as tfq

import cirq
import sympy
import numpy as np
from numpy import pi

class VQC(layers.Layer):
    def __init__(self, units):
        super(VQC, self).__init__()
        self.parameters = tf.Variable(
            np.random.uniform(0, 2 * np.pi, (1, 3 * units)),
            name="Q_param",
            dtype=tf.float32,
        )

        qubits, model_circuit = self.blueprint(units)
        observables = [ cirq.Z(bit) for bit in qubits ]
        self.layer = tfq.layers.ControlledPQC(model_circuit, operators=observables)
        self.flatten = layers.Flatten()

    def blueprint(self, units):
        qubits = cirq.LineQubit.range(units)
        circuit = cirq.Circuit()

        # Angle Encoding
        embedding = sympy.symbols(f'e0:{units}')
        for i in range(units):
            circuit.append(cirq.H(qubits[i]))
            circuit.append(cirq.Z(qubits[i])**embedding[i])

        # CNOT gates
        if units > 1:
            for i in range(units):
                circuit.append(cirq.CX(qubits[i], qubits[(i + 1) % units]))

        # Final rotations
        alpha = sympy.symbols(f'a0:{units}')
        beta = sympy.symbols(f'b0:{units}')
        gamma = sympy.symbols(f'g0:{units}')
        for i in range(units):
            circuit.append(cirq.X(qubits[i]) ** alpha[i])
            circuit.append(cirq.Y(qubits[i]) ** beta[i])
            circuit.append(cirq.Z(qubits[i]) ** gamma[i])

        return qubits, circuit
    
    def call(self, x):
        s = tf.shape(x)
        x1 = self.flatten(x)
        x2 = tf.tile(self.parameters, tf.stack([tf.shape(x)[0], 1]))
        x = tf.concat([x1, x2], -1)
        empty_circuit = tf.tile(tfq.convert_to_tensor([cirq.Circuit()]), tf.stack([tf.shape(x)[0]]))
        x = self.layer([empty_circuit, x])
        return tf.reshape(x, s)

class QLSTMCell(layers.LSTMCell):

    def __init__(self, units, **kwargs):
        super(layers.LSTMCell, self).__init__(units, implementation=2, **kwargs)
        self.vqc_i = VQC(units)
        self.vqc_f = VQC(units)
        self.vqc_c = VQC(units)
        self.vqc_o = VQC(units)

    """
        Override kera's LSTM computing method.
    """
    def _compute_carry_and_output_fused(self, z, c_tm1):
        z0, z1, z2, z3 = z
        i = self.recurrent_activation(self.vqc_i(z0))
        f = self.recurrent_activation(self.vqc_f(z1))
        c = f * c_tm1 + i * self.activation(self.vqc_c(z2))
        o = self.recurrent_activation(self.vqc_o(z3))
        return c, o

class QLSTMEncoder(layers.Layer):
    def __init__(self, vocab_size: int, maxlen: int, embed_dim: int, num_blocks: int):
        super(QLSTMEncoder, self).__init__()
        self.embed = tf.keras.models.Sequential([
            layers.Input(shape=(maxlen,)),
            layers.Embedding(input_dim=vocab_size, output_dim=embed_dim),
        ])
        self.layers = []
        for _ in range(num_blocks):
            self.layers.append(layers.RNN(
                                QLSTMCell(embed_dim),
                                return_sequences=True,
                                return_state=True,))

    def call(self, x):
        x = self.embed(x)
        for layer in self.layers:
            whole_seq_output, final_memory_state, final_carry_state = layer(x)
            x = whole_seq_output
        return x
