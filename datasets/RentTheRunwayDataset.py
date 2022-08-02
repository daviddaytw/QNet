import tensorflow as tf
import json
import tensorflow_datasets.public_api as tfds

_CITATION = """\
@inproceedings{10.1145/3240323.3240398,
    author = {Misra, Rishabh and Wan, Mengting and McAuley, Julian},
    title = {Decomposing Fit Semantics for Product Size Recommendation in Metric Spaces},
    year = {2018},
    isbn = {9781450359016},
    publisher = {Association for Computing Machinery},
    address = {New York, NY, USA},
    url = {https://doi.org/10.1145/3240323.3240398},
    doi = {10.1145/3240323.3240398},
    abstract = {Product size recommendation and fit prediction are critical in order to improve customers' shopping experiences and to reduce product return rates. Modeling customers' fit feedback is challenging due to its subtle semantics, arising from the subjective evaluation of products, and imbalanced label distribution. In this paper, we propose a new predictive framework to tackle the product fit problem, which captures the semantics behind customers' fit feedback, and employs a metric learning technique to resolve label imbalance issues. We also contribute two public datasets collected from online clothing retailers.},
    booktitle = {Proceedings of the 12th ACM Conference on Recommender Systems},
    pages = {422-426},
    numpages = {5},
    location = {Vancouver, British Columbia, Canada},
    series = {RecSys '18}
}
"""
_HOMEPAGE = "http://deepx.ucsd.edu/public/jmcauley/renttherunway/renttherunway_final_data.json.gz"
_URL = 'http://deepx.ucsd.edu/public/jmcauley/renttherunway/renttherunway_final_data.json.gz'

class RentTheRunwayDataset(tfds.core.GeneratorBasedBuilder):
  VERSION = tfds.core.Version('0.1.1')

  def _info(self):
    return tfds.core.DatasetInfo(
        builder=self,
        description=('clothing fit feedback'),
        features=tfds.features.FeaturesDict({
            "text": tfds.features.Text(),
            "label": tfds.features.Scalar(dtype=tf.int32),
        }),
        supervised_keys=('text', 'label'),
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
            gen_kwargs={ "filepath": path },
        ),
    ]

  def _generate_examples(self, filepath):
    with open(filepath) as f:
        idx = 0
        for line in f.readlines():
            row = json.loads(line)
            if row['rating'] == None:
                continue
            idx += 1
            yield idx, {'text': row['review_summary'], 'label': row['rating']}
