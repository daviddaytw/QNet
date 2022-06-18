import tensorflow as tf
import tensorflow_datasets as tfds
from . import ColBERTDataset, StackOverflowDataset

mapping = {
    'colbert': 'ColBERTDataset',
    'stackoverflow': 'StackOverflowDataset',
}


output_size = {
    'colbert': 2,
    'stackoverflow': 20,
}

def get_dataset(dataset: str, train_ratio: int=0.83):
    ds = tfds.load(mapping[dataset], split='train', as_supervised=True)\
          .shuffle(buffer_size=1000)\
          .map(lambda x, y: (x, tf.one_hot(tf.cast(y, tf.int32), output_size[dataset])))
    return ds.take(int(len(ds) * train_ratio)), ds.skip(int(len(ds) * train_ratio))

def get_dataset_output_size(dataset: str):
    return output_size[dataset]
