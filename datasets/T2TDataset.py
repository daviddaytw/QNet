import tensorflow_datasets as tfds
tfds.core.utils.gcs_utils._is_gcs_disabled = True

import random

from Bio import SeqIO
from tqdm import tqdm

# reproducible randomness
random.seed(42)

_CITATION = """
"""
_HOMEPAGE = "https://www.ncbi.nlm.nih.gov/assembly/GCA_009914755.4/"
_URL = 'https://ftp.ncbi.nlm.nih.gov/genomes/refseq/vertebrate_mammalian/Homo_sapiens/all_assembly_versions/GCF_009914755.1_T2T-CHM13v2.0/GCF_009914755.1_T2T-CHM13v2.0_genomic.fna.gz'

def _generate_documents(chr_sequence, sentences_bounds=(50, 100), lenghts_bounds=(500, 1000)):
        """
        From a single chromosome yield a set of documents that cover that chromosome.
        This operation is done ten-fold.
        """
        C = len(chr_sequence)  # chromosome length

        for _ in range(10):
            q = random.randint(0, 5000)  # random start position from the 5' end
            while q < C:
                s = random.randint(*sentences_bounds)  # number of sentences per document
                d = []
                for _ in range(s):
                    l = random.randint(*lenghts_bounds)  # length of each sentence
                    d.append(str(chr_sequence[q : q + l]).upper())
                    q += l  # update position for the new sentence
                yield d

class T2TDataset(tfds.core.GeneratorBasedBuilder):
  VERSION = tfds.core.Version('0.1.1')

  def _info(self):
    return tfds.core.DatasetInfo(
        builder=self,
        description=(''),
        features=tfds.features.FeaturesDict({
            "text": tfds.features.Text(),
        }),
        homepage=_HOMEPAGE,
        citation=_CITATION,
    )

  def _split_generators(self, dl_manager):
    path = dl_manager.download_and_extract(_URL)

    return [
        # only use TRAIN name if source data is non-split
        # https://tensorflow.google.cn/datasets/add_dataset?hl=zh-cn#%E6%8C%87%E5%AE%9A%E6%95%B0%E6%8D%AE%E9%9B%86%E5%88%86%E5%89%B2
        tfds.core.SplitGenerator(
            name=tfds.Split.TRAIN,
            gen_kwargs={ "path": path },
        ),
    ]

  def _generate_examples(self, path):
    i = 0
    for record in tqdm(SeqIO.parse(path, "fasta")):
        if "mitochondrion" not in record.description:
            for document in tqdm(_generate_documents(record.seq), desc=record.description):
                yield i, {'text': "\n".join(document) + "\n"}
                i += 1

    print(i)

for i in tfds.load('T2TDataset'):
    print(i)