import tensorflow as tf
import pennylane as qml
from pennylane import numpy as np

def encode_embedding(srcs, embed_size, wires):
    emb_size = embed_size * 3
    seq_len = srcs.shape[-1] // emb_size
    for i in range(seq_len):
        emb = srcs[:, i * emb_size : (i + 1) * emb_size]
        token_wires = wires[i* embed_size: (i + 1) * embed_size]
        qml.AngleEmbedding(features=emb[:, :embed_size], wires=token_wires, rotation='X')
        qml.AngleEmbedding(features=emb[:, embed_size:embed_size*2], wires=token_wires, rotation='Y')
        qml.AngleEmbedding(features=emb[:, embed_size*2:], wires=token_wires, rotation='Z')

def quanttention(weights, wires, embed_size):
    for idx in range(0, len(wires), embed_size):
        token_wires = wires[idx : idx + embed_size]
        qml.StronglyEntanglingLayers(weights=weights[0], wires=token_wires)

    for idx in range(embed_size):
        embed_wires = [ wires[i + idx] for i in range(0, len(wires), embed_size) ]
        qml.QFT(wires=embed_wires)

def feedforward(weights, wires, embed_size):
    for idx in range(0, len(wires), embed_size):
        token_wires = wires[idx : idx + embed_size]
        
        qml.StronglyEntanglingLayers(weights=weights[0], wires=token_wires)
        qml.templates.GroverOperator(wires=token_wires)
        qml.StronglyEntanglingLayers(weights=weights[1], wires=token_wires)

def encoder(weights, encoder_wires, embed_size):
    quanttention(weights[:1], wires=encoder_wires, embed_size=embed_size)
    feedforward(weights[1:], wires=encoder_wires, embed_size=embed_size)

def decoder(weights, decoder_wires, encoder_wires):
    quanttention(weights[0:1], wires=decoder_wires)
    
    for idx, tgt_wire in enumerate(decoder_wires):
        for src_wire in range(idx % num_embed_wire, len(encoder_wires), num_embed_wire):
            qml.CNOT(wires=[src_wire, tgt_wire])
    
    quanttention(weights[1:2], wires=decoder_wires)
    feedforward(weights[2:4], wires=decoder_wires)

def QNetEncoder(
        embed_size : int,
        src_seq_len : int,
        num_blocks : int = 1,
    ):
    n_wires = embed_size * src_seq_len
    dev = qml.device('default.qubit.tf', wires=n_wires)
    
    @qml.qnode(dev, interface="tf", diff_method="backprop")
    def berq_circuit(inputs, weights):
        embed_size = weights.shape[2]
        src_seq_len = inputs.shape[-1] // ( embed_size * 3 )

        wires = range(embed_size * src_seq_len)

        # Encoder input word embedding
        encode_embedding(inputs, embed_size, wires)

        # Repeat encoder N times
        for block_idx in range(0, weights.shape[0], 3):
            encoder(weights[block_idx : block_idx + 3], wires, embed_size)

        return [ qml.expval(qml.PauliZ(i)) for i in wires ]

    weight_shapes = {
        "weights": (3 * num_blocks, 1, embed_size, 3)
    }

    return  qml.qnn.KerasLayer(berq_circuit, weight_shapes, batch_idx = 0, output_dim = n_wires)

def QNet(
        embed_size : int,
        src_seq_len : int,
        tgt_seq_len : int,
        tgt_vocab_size : int,
        num_blocks : int = 1,
    ):
    n_wires = (embed_size) * (src_seq_len + tgt_seq_len)
    dev = qml.device('default.qubit.tf', wires=n_wires)

    @qml.qnode(dev, interface="tf", diff_method="backprop")
    def qnet_circuit(inputs, weights_encoder, weights_decoder, weights_aux):
        embed_size = weights_encoder.shape[2]
        src_seq_len, tgt_seq_len = weights_aux.shape

        encoder_wires = range(embed_size * src_seq_len)
        decoder_wires = [ i + encoder_wires[-1] + 1 for i in range(embed_size * tgt_seq_len) ]

        srcs = inputs[:, :src_seq_len]
        tgts = inputs[:, src_seq_len:]

        # Encoder input word embedding
        encode_embedding(srcs, encoder_wires)

        # Repeat encoder N times
        for block_idx in range(0, weights_encoder.shape[0], 3):
            encoder(weights_encoder[block_idx : block_idx + 3], encoder_wires)

        # Encoder output word embedding (shifted-right)
        encode_embedding(tgts, decoder_wires)

        # Repeat Decoder N times
        for block_idx in range(0, weights_decoder.shape[0], 4):
            decoder(weights_decoder[block_idx : block_idx + 4], decoder_wires, encoder_wires)

        return [ qml.expval(qml.PauliZ(i)) for i in decoder_wires ]

    weight_shapes = {
        "weights_encoder": (3 * num_blocks, 1, embed_size, 3),
        "weights_decoder": (4 * num_blocks, 1, embed_size, 3),
        "weights_aux" : ( src_seq_len, tgt_seq_len ),
    }

    qmodel = qml.qnn.KerasLayer(qnet_circuit, weight_shapes, output_dim = tgt_seq_len)
    cmodel = tf.keras.layers.Dense(tgt_vocab_size)
    model = tf.keras.models.Sequential([qmodel, cmodel])
    return model

if __name__ == '__main__':
    num_embed_wire = 2
    src_token_len = 1
    tgt_token_len = 1
    n_wires = (num_embed_wire) * (src_token_len + tgt_token_len)
    dev = qml.device('default.qubit', wires=n_wires)
    qnode = qml.QNode(circuit, dev)

    srcs = np.random.random(size=(src_token_len, num_embed_wire))
    tgts = np.random.random(size=(tgt_token_len, num_embed_wire))
    encoderWeight = np.random.random(size=(3, 1, num_embed_wire, 3))
    decoderWeight = np.random.random(size=(4, 1, num_embed_wire, 3))
    
    print(qml.draw(qnode)(srcs, tgts, encoderWeight, decoderWeight))
