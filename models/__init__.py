import tensorflow as tf
from .FNet import FNet
from .QNet import QNetEncoder
from .QLSTM import QLSTMEncoder
from .Transformer import TransformerEncoder

mapping = {
    'transformer': TransformerEncoder,
    'qnet': QNetEncoder,
    'qlstm' : QLSTMEncoder,
    'fnet': FNet,
}

def get_model(args, vocab_size: int):
    model = mapping[args.model]
    if args.model == 'qnet':
        instance = model(vocab_size, args.seq_len, args.embed_size, args.num_blocks, args.qnet_depth)
    else:
        instance = model(vocab_size, args.seq_len, args.embed_size, args.num_blocks)

    # Build model with one inference
    instance(tf.ones((1, args.seq_len), dtype=tf.int64))

    # Count parameters
    total_parameters = 0
    for variable in instance.variables:
        if 'embedding' in variable.name:
            continue
        # shape is an array of tf.Dimension
        shape = variable.get_shape()
        variable_parameters = 1
        for dim in shape:
            variable_parameters *= dim
        total_parameters += variable_parameters
    print('Number of Parameters w/o embedding layer:', total_parameters)

    return instance

def list_model():
    return mapping.keys()