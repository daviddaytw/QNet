from .FNet import FNet
from .QNet import QNetEncoder
from .Transformer import TransformerEncoder

mapping = {
    'transformer': TransformerEncoder,
    'qnet': QNetEncoder,
    'fnet': FNet,
}

def get_model(model: str, vocab_size: int, embed_size: int, seq_len: int, num_blocks: int):
    return mapping[model](vocab_size, seq_len, embed_size, num_blocks)
