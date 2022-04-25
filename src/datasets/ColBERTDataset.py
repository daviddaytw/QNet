import sys
import os
import string
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow_datasets.public_api as tfds

MAX_LEN=7
FILTER='"&(),-/:;<=>[\\]_`{|}~\t\n0123456789' or string.punctuation

_CITATION = """\
@misc{https://doi.org/10.48550/arxiv.2004.12832,
    doi = {10.48550/ARXIV.2004.12832},
    url = {https://arxiv.org/abs/2004.12832},
    author = {Khattab, Omar and Zaharia, Matei},
    keywords = {Information Retrieval (cs.IR), Computation and Language (cs.CL), FOS: Computer and information sciences, FOS: Computer and information sciences},
    title = {ColBERT: Efficient and Effective Passage Search via Contextualized Late Interaction over BERT},
    publisher = {arXiv},
    year = {2020},
    copyright = {arXiv.org perpetual, non-exclusive license}
}
"""
_HOMEPAGE = "https://github.com/Moradnejad/ColBERT-Using-BERT-Sentence-Embedding-for-Humor-Detection"
_URL = 'https://raw.githubusercontent.com/Moradnejad/ColBERT-Using-BERT-Sentence-Embedding-for-Humor-Detection/master/Data/dataset.csv'

class ColBERTDataset(tfds.core.GeneratorBasedBuilder):
  VERSION = tfds.core.Version('0.1.0')

  def _info(self):
    return tfds.core.DatasetInfo(
        builder=self,
        description=('200k-short-text-for-humor-detect'),
        features=tfds.features.FeaturesDict({
            "text": tfds.features.Tensor(dtype=tf.float64, shape=(MAX_LEN,)),
            "label": tfds.features.ClassLabel(names=['True', 'False']),
        }),
        supervised_keys=('text', 'label'),
        homepage=_HOMEPAGE,
        citation=_CITATION,
    )

  def _split_generators(self, dl_manager):
    path = dl_manager.download(_URL)

    return [
        # only use TRAIN name if source data is non-split
        # https://tensorflow.google.cn/datasets/add_dataset?hl=zh-cn#%E6%8C%87%E5%AE%9A%E6%95%B0%E6%8D%AE%E9%9B%86%E5%88%86%E5%89%B2
        tfds.core.SplitGenerator(
            name=tfds.Split.TRAIN,
            gen_kwargs={ "filepath": path },
        ),
    ]

  def _generate_examples(self, filepath):
    df = pd.read_csv(filepath)
    # df = df.sample(frac=1, random_state=42).reset_index(drop=True)

    tokenizer = tf.keras.preprocessing.text.Tokenizer(filters=FILTER, lower=True, split=' ')

    texts = np.array(df['text'])
    tokenizer.fit_on_texts(texts)

    texts = tokenizer.texts_to_sequences(df['text'])
    if MAX_LEN > 0:
        texts = tf.keras.preprocessing.sequence.pad_sequences(texts, maxlen=MAX_LEN, dtype='float', padding='post', truncating='post', value=0.0)

    labels = tf.keras.utils.to_categorical(df['humor'], num_classes=2, dtype='int8')

    for i in range(len(labels)):
        yield i, {'text': texts[i], 'label': labels[i]}