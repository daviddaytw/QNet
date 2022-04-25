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
@inproceedings{xu-etal-2015-short,
    title = "Short Text Clustering via Convolutional Neural Networks",
    author = "Xu, Jiaming  and
      Wang, Peng  and
      Tian, Guanhua  and
      Xu, Bo  and
      Zhao, Jun  and
      Wang, Fangyuan  and
      Hao, Hongwei",
    booktitle = "Proceedings of the 1st Workshop on Vector Space Modeling for Natural Language Processing",
    month = jun,
    year = "2015",
    address = "Denver, Colorado",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/W15-1509",
    doi = "10.3115/v1/W15-1509",
    pages = "62--69",
}
"""
_HOMEPAGE = "https://github.com/jacoxu/StackOverflow"
_URL=[
    'https://raw.githubusercontent.com/jacoxu/StackOverflow/master/rawText/title_StackOverflow.txt',
    'https://raw.githubusercontent.com/jacoxu/StackOverflow/master/rawText/label_StackOverflow.txt'
]
LABEL= {
    1: 'wordpress',
    2: 'oracle',
    3: 'svn',
    4: 'apache',
    5: 'excel',
    6: 'matlab',
    7: 'visual-studio',
    8: 'cocoa',
    9: 'osx',
    10: 'bash',
    11: 'spring',
    12: 'hibernate',
    13: 'scala',
    14: 'sharepoint',
    15: 'ajax',
    16: 'qt',
    17: 'drupal',
    18: 'linq',
    19: 'haskell',
    20: 'magento',
}

class StackOverflowDataset(tfds.core.GeneratorBasedBuilder):
  VERSION = tfds.core.Version('0.1.0')

  def _info(self):
    return tfds.core.DatasetInfo(
        builder=self,
        description=('stack_overflow-title-classification'),
        features=tfds.features.FeaturesDict({
            "text": tfds.features.Tensor(dtype=tf.float64, shape=(MAX_LEN,)),
            "label": tfds.features.Tensor(dtype=tf.int8, shape=(len(LABEL),)),
        }),
        supervised_keys=('text', 'label'),
        homepage=_HOMEPAGE,
        citation=_CITATION,
    )

  def _split_generators(self, dl_manager):
    title_path = tf.keras.utils.get_file('stack_overflow-title', _URL[0])
    label_path = tf.keras.utils.get_file('stack_overflow-label', _URL[1])

    return [
        # only use TRAIN name if source data is non-split
        # https://tensorflow.google.cn/datasets/add_dataset?hl=zh-cn#%E6%8C%87%E5%AE%9A%E6%95%B0%E6%8D%AE%E9%9B%86%E5%88%86%E5%89%B2
        tfds.core.SplitGenerator(
            name=tfds.Split.TRAIN,
            gen_kwargs={ "text_filepath": title_path, "label_filepath": label_path },
        ),
    ]

  def _generate_examples(self, text_filepath, label_filepath):
    df = pd.DataFrame(columns=['text', 'label'])
    df['text'] = pd.read_fwf(text_filepath, header=None, on_bad_lines='skip')[0]
    df['label'] = pd.read_fwf(label_filepath, header=None, on_bad_lines='skip')[0]
    # df = df.sample(frac=1, random_state=42).reset_index(drop=True)

    tokenizer = tf.keras.preprocessing.text.Tokenizer(filters=FILTER, lower=True, split=' ')

    texts = np.array(df['text'])
    tokenizer.fit_on_texts(texts)

    texts = tokenizer.texts_to_sequences(df['text'])
    if MAX_LEN > 0:
        texts = tf.keras.preprocessing.sequence.pad_sequences(texts, maxlen=MAX_LEN, dtype='float', padding='post', truncating='post', value=0.0)

    labels = tf.keras.utils.to_categorical(df['label'] - 1, num_classes=len(LABEL), dtype='int8')

    for i in range(len(labels)):
        yield i, {'text': texts[i], 'label': labels[i]}