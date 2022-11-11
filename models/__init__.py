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

def count_params(args):
    model = get_model(args, 10) # 10 is a vocab size

    # Build model with one inference
    model(tf.ones((1, args.seq_len), dtype=tf.int64))

    total_parameters = 0
    for variable in model.variables:
        if 'embedding' in variable.name:
            continue
        # shape is an array of tf.Dimension
        shape = variable.get_shape()
        variable_parameters = 1
        for dim in shape:
            variable_parameters *= dim
        total_parameters += variable_parameters
    return total_parameters

def get_model(args, vocab_size: int):
    model = mapping[args.model]
    if args.model == 'qnet':
        return model(vocab_size, args.seq_len, args.embed_size, args.num_blocks, args.qnet_depth)
    else:
        return model(vocab_size, args.seq_len, args.embed_size, args.num_blocks)

def list_model():
    return mapping.keys()