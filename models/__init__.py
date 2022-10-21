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
        return model(vocab_size, args.seq_len, args.embed_size, args.num_blocks, args.qnet_depth)
    else:
        return model(vocab_size, args.seq_len, args.embed_size, args.num_blocks)

def list_model():
    return mapping.keys()