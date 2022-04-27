import sys
import os
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow_datasets.public_api as tfds

MAX_LEN=7

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
    '0': 'wordpress',
    '1': 'oracle',
    '2': 'svn',
    '3': 'apache',
    '4': 'excel',
    '5': 'matlab',
    '6': 'visual-studio',
    '7': 'cocoa',
    '8': 'osx',
    '9': 'bash',
    '10': 'spring',
    '11': 'hibernate',
    '12': 'scala',
    '13': 'sharepoint',
    '14': 'ajax',
    '15': 'qt',
    '16': 'drupal',
    '17': 'linq',
    '18': 'haskell',
    '19': 'magento',
}

class StackOverflowDataset(tfds.core.GeneratorBasedBuilder):
  VERSION = tfds.core.Version('0.1.1')

  def _info(self):
    return tfds.core.DatasetInfo(
        builder=self,
        description=('stack_overflow-title-classification'),
        features=tfds.features.FeaturesDict({
            "text": tfds.features.Text(),
            "label": tfds.features.ClassLabel(names=list(LABEL.keys())),
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
    df['label'] = pd.read_fwf(label_filepath, header=None, on_bad_lines='skip')[0] - 1

    for i in range(len(df['label'])):
        yield i, {'text': df['text'][i], 'label': df['label'][i]}