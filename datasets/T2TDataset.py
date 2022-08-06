import tensorflow_datasets as tfds
tfds.core.utils.gcs_utils._is_gcs_disabled = True

import random

from Bio import SeqIO
from tqdm import tqdm

# reproducible randomness
random.seed(42)

_CITATION = """
Miga KH, Koren S, Rhie A, Vollger MR, Gershman A, Bzikadze A, Brooks S, Howe E, Porubsky D, Logsdon GA, Schneider VA,
Potapova T, Wood J, Chow W, Armstrong J, Fredrickson J, Pak E, Tigyi K, Kremitzki M, Markovic C, Maduro V, Dutra A,
Bouffard GG, Chang AM, Hansen NF, Wilfert AB, Thibaud-Nissen F, Schmitt AD, Belton JM, Selvaraj S, Dennis MY, Soto
DC, Sahasrabudhe R, Kaya G, Quick J, Loman NJ, Holmes N, Loose M, Surti U, Risques RA, Graves Lindsay TA, Fulton R,
Hall I, Paten B, Howe K, Timp W, Young A, Mullikin JC, Pevzner PA, Gerton JL, Sullivan BA, Eichler EE, Phillippy AM.
Telomere-to-telomere assembly of a complete human X chromosome. Nature. 2020 Sep;585(7823):79-84. doi: 10.1038/
s41586-020-2547-7. Epub 2020 Jul 14. PMID: 32663838; PMCID: PMC7484160.
"""
_HOMEPAGE = "https://www.ncbi.nlm.nih.gov/assembly/GCA_009914755.4/"
_URL = 'https://ftp.ncbi.nlm.nih.gov/genomes/refseq/vertebrate_mammalian/Homo_sapiens/all_assembly_versions/GCF_009914755.1_T2T-CHM13v2.0/GCF_009914755.1_T2T-CHM13v2.0_genomic.fna.gz'

MAX_LEN = 16

class T2TDataset(tfds.core.GeneratorBasedBuilder):
  VERSION = tfds.core.Version('0.1.2')

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

  # 24 * 100000 texts
  def _generate_examples(self, path):
    i = 0
    for record in SeqIO.parse(path, "fasta"):
        q = random.randint(0, 5000)
        for _ in range(100000):
            yield i, {'text': str(record.seq[q : q + MAX_LEN])}
            q += MAX_LEN
            i += 1