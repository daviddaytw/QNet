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

def get_dataset(dataset: str, batch_size: int, train_ratio: int=0.83):
    ds = tfds.load(mapping[dataset], split='train', as_supervised=True)\
             .map(lambda x, y: (x, tf.one_hot(tf.cast(y, tf.int32), output_size[dataset])))

    train_data = ds.take(int(len(ds) * train_ratio))\
                   .shuffle(1000)\
                   .batch(batch_size)

    test_data = ds.skip(int(len(ds) * train_ratio))\
                  .batch(batch_size)

    return train_data, test_data

def get_dataset_output_size(dataset: str):
    return output_size[dataset]
