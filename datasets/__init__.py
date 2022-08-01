import tensorflow as tf
import tensorflow_datasets as tfds
from . import ColBERTDataset, StackOverflowDataset, T2TDataset

mapping = {
    'colbert': 'ColBERTDataset',
    'stackoverflow': 'StackOverflowDataset',
    't2t': 'T2TDataset',
}


output_size = {
    'colbert': 2,
    'stackoverflow': 20,
}

def get_dataset(dataset: str, batch_size: int, train_ratio: int=0.83):
    # for classification
    if dataset in ['colbert', 'stackoverflow']:
        ds = tfds.load(mapping[dataset], split='train', as_supervised=True)
        if output_size[dataset] > 2:
            ds = ds.map(lambda x, y: (x, tf.one_hot(tf.cast(y, tf.int32), output_size[dataset])))

        train_data = ds.take(int(len(ds) * train_ratio))\
                    .shuffle(1000)\
                    .batch(batch_size)

        test_data = ds.skip(int(len(ds) * train_ratio))\
                    .batch(batch_size)

        return train_data, test_data
    # for mask LM
    elif dataset in ['t2t']:
        all_data = tfds.load('T2TDataset', split='train')\
                       .shuffle(1000)
        return all_data, None

def get_dataset_output_size(dataset: str):
    if output_size[dataset] > 2:
        return output_size[dataset]
    else:
        return 1
