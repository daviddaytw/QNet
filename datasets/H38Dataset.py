import tensorflow_datasets as tfds
tfds.core.utils.gcs_utils._is_gcs_disabled = True

import random
from Bio import SeqIO

# reproducible randomness
random.seed(42)

_CITATION = """
"""
_HOMEPAGE = "https://github.com/AIRI-Institute/GENA_LM"
_URL = 'https://raw.githubusercontent.com/AIRI-Institute/GENA_LM/main/downstream_tasks/promoter_prediction/hg38_len_300.fa.txt'

MAX_LEN = 16

def sample_negative(sequence):
    n = len(sequence)
    step = n // 4
    subs = [ sequence[i:i+step] for i in range(0, n, step) ]
    random.shuffle(subs)
    return ''.join(subs)

class H38Dataset(tfds.core.GeneratorBasedBuilder):
  VERSION = tfds.core.Version('0.1.1')

  def _info(self):
    return tfds.core.DatasetInfo(
        builder=self,
        description=(''),
        features=tfds.features.FeaturesDict({
            "text": tfds.features.Text(),
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
            gen_kwargs={ "path": path },
        ),
    ]


  def _generate_examples(self, path):
    sentences = []
    for record in SeqIO.parse(path, "fasta"):
        if len(sentences) >= 25600:
            break
        sentences.append(str(record.seq[:MAX_LEN]))

    for i, text in enumerate(sentences):
        yield i, {'text': text, 'label': False}

    for i, text in enumerate(sentences, len(sentences)):
        yield i, {'text': sample_negative(text), 'label': True}