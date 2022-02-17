import argparse
from torchtext import datasets
from models.transformer import TextClassifier as transformerClassifier
from models.fformer import TextClassifier as fformerClassifier
from models.qfformer import TextClassifier as qfformerClassifier
from models.fnet import TextClassifier as fnetClassifier
from models.flatnet import TextClassifier as flatnetClassifier
from models.qnet import TextClassifier as qnetClassifier
from models.qfnet import TextClassifier as qfnetClassifier


parser = argparse.ArgumentParser()
parser.add_argument('-B', '--batch_size', default=64, type=int)
parser.add_argument('-E', '--n_epochs', default=15, type=int)
parser.add_argument('-l', '--lr', default=0.001, type=float)
parser.add_argument('-e', '--embed_dim', default=16, type=int)
parser.add_argument('-s', '--max_seq_len', default=256, type=int)
parser.add_argument('-f', '--ffn_dim', default=16, type=int)
parser.add_argument('-t', '--n_transformer_blocks', default=1, type=int)
parser.add_argument('-H', '--n_heads', default=2, type=int)
parser.add_argument('-d', '--dropout_rate', default=0.1, type=float)
args = parser.parse_args()

models = {
    'Transformer' : transformerClassifier,
    'fformer' : fformerClassifier,
    'qfformer' : qfformerClassifier,
    'fnet': fnetClassifier,
    'qnet': qnetClassifier,
    'qfnet': qfnetClassifier,
    'FlatNet': flatnetClassifier,
}

datasets = {
    'IMDB': datasets.IMDB,
    'AG_NEWS': datasets.AG_NEWS,
    'DBpedia': datasets.DBpedia,
}
