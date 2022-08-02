import pandas as pd
import tensorflow_datasets.public_api as tfds

_CITATION = """\
@inproceedings{chakraborty2016stop,
  title={Stop Clickbait: Detecting and preventing clickbaits in online news media},
  author={Chakraborty, Abhijnan and Paranjape, Bhargavi and Kakarla, Sourya and Ganguly, Niloy},
  booktitle={Advances in Social Networks Analysis and Mining (ASONAM), 2016 IEEE/ACM International Conference on},
  pages={9--16},
  year={2016},
  organization={IEEE}
}
"""
_HOMEPAGE = "https://github.com/bhargaviparanjape/clickbait"
_URL = 'https://github.com/owais4321/clickbait-classification/raw/master/clickbait_data.csv'

class ClickbaitDataset(tfds.core.GeneratorBasedBuilder):
  VERSION = tfds.core.Version('0.1.1')

  def _info(self):
    return tfds.core.DatasetInfo(
        builder=self,
        description=('16000 clickbaits in online news media'),
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
            gen_kwargs={ "filepath": path },
        ),
    ]

  def _generate_examples(self, filepath):
    df = pd.read_csv(filepath)

    for i in range(len(df['headline'])):
        yield i, {'text': df['headline'][i], 'label': df['clickbait'][i]}