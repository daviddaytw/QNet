import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow_datasets.public_api as tfds

_CITATION = """\
@inproceedings{levow-2006-third,
    title = "The Third International {C}hinese Language Processing Bakeoff: Word Segmentation and Named Entity Recognition",
    author = "Levow, Gina-Anne",
    booktitle = "Proceedings of the Fifth {SIGHAN} Workshop on {C}hinese Language Processing",
    month = jul,
    year = "2006",
    address = "Sydney, Australia",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/W06-0115",
    pages = "108--117",
}
"""
_HOMEPAGE = "https://github.com/OYE93/Chinese-NLP-Corpus"
_URL=[
    'https://raw.githubusercontent.com/OYE93/Chinese-NLP-Corpus/master/NER/MSRA/msra_train_bio.txt',
    'https://raw.githubusercontent.com/OYE93/Chinese-NLP-Corpus/master/NER/MSRA/msra_test_bio.txt'
]
LABEL= {
    'O': 0,
    'B-ORG': 1,
    'I-ORG': 2,
    'B-PER': 3,
    'I-PER': 4,
    'B-LOC': 5,
    'I-LOC': 6,
}
ID_TO_LABEL= dict(zip(LABEL.values(), LABEL.keys()))
MAX_LEN = 8

class MSRADataset(tfds.core.GeneratorBasedBuilder):
  VERSION = tfds.core.Version('0.1.2')

  def _info(self):
    return tfds.core.DatasetInfo(
        builder=self,
        description=('stack_overflow-title-classification'),
        features=tfds.features.FeaturesDict({
            "text": tfds.features.Tensor(shape=(MAX_LEN,), dtype=tf.string),
            "label": tfds.features.Tensor(shape=(MAX_LEN,), dtype=tf.int64),
        }),
        metadata=tfds.core.MetadataDict({ 'id_to_label': ID_TO_LABEL }),
        supervised_keys=('text', 'label'),
        homepage=_HOMEPAGE,
        citation=_CITATION,
    )

  def _split_generators(self, dl_manager):
    train_path = dl_manager.download(_URL[0])
    tesst_path = dl_manager.download(_URL[1])

    return [
        # only use TRAIN name if source data is non-split
        # https://tensorflow.google.cn/datasets/add_dataset?hl=zh-cn#%E6%8C%87%E5%AE%9A%E6%95%B0%E6%8D%AE%E9%9B%86%E5%88%86%E5%89%B2
        tfds.core.SplitGenerator(
            name=tfds.Split.TRAIN,
            gen_kwargs={ "paths": [train_path, tesst_path] },
        ),
    ]

  def _generate_examples(self, paths):
    files = map(lambda x: pd.read_csv(x, sep='\t', names=['text', 'label']), paths)
    df = pd.concat([*files]).dropna().reset_index(drop=True)

    texts = df['text'].to_numpy()
    texts = texts[:len(texts) - len(texts) % MAX_LEN].reshape((-1, MAX_LEN))

    labels = np.vectorize(LABEL.get)(df['label'].to_numpy())
    labels = labels[:len(labels) - len(labels) % MAX_LEN].reshape((-1, MAX_LEN))

    for i in range(len(labels)):
        yield i, {'text': texts[i], 'label': labels[i]}