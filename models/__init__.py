from .FNet import FNet
from .QNet import QNetEncoder
from .Transformer import TransformerEncoder

mapping = {
    'transformer': TransformerEncoder,
    'qnet': QNetEncoder,
    'fnet': FNet,
}

def get_model(args, vocab_size: int):
    model = mapping[args.model]
    if args.model == 'qnet':
        return model(vocab_size, args.seq_len, args.embed_size, args.num_blocks, args.qnet_depth)
    else:
        return model(vocab_size, args.seq_len, args.embed_size, args.num_blocks)
