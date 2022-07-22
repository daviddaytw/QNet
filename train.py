# since we encounter this [problem](https://colab.research.google.com/github/tensorflow/docs/blob/master/site/en/tutorials/distribute/multi_worker_with_keras.ipynb?hl=id-IDCache#scrollTo=Mhq3fzyR5hTw)
# it must put on the file starts before initialize other class and after set env.
from utils.args_parser import solve_args
args = solve_args(multi_worker_strategy=True)

from utils.distributed_train import MultiWorkerStrategy

import os, time, json

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
from datasets import get_dataset, get_dataset_output_size
from models import get_model


@MultiWorkerStrategy
def train(train_data, test_data):
    vectorize_layer = layers.TextVectorization(
        standardize="lower_and_strip_punctuation",
        output_mode='int',
        output_sequence_length=args.seq_len
    )

    train_text = train_data.flat_map(lambda text, label: tf.data.Dataset.from_tensor_slices(text))
    vectorize_layer.adapt(train_text)
    vocab_size = vectorize_layer.vocabulary_size()
    print('Vocab size:', vocab_size)

    model = tf.keras.models.Sequential([
        tf.keras.Input(shape=(1,), dtype=tf.string),
        vectorize_layer,
        get_model(
            args.model,
            vocab_size,
            args.embed_size,
            args.seq_len,
            args.num_blocks,
        ),
        layers.GlobalAveragePooling1D(),
        layers.Dense(get_dataset_output_size(args.dataset)),
    ])

    lr_decayed_fn = tf.keras.optimizers.schedules.CosineDecay(args.lr, args.epochs * len(train_data), alpha=1e-2)
    opt = tf.keras.optimizers.Adam(learning_rate=lr_decayed_fn, beta_1=0.9, beta_2=0.98, epsilon=1e-09)
    model.compile(
        opt,
        loss=tf.keras.losses.CategoricalCrossentropy(),
        metrics=["accuracy"]
    )

    print(model.summary())
    fitting = model.fit(
                train_data,
                batch_size=args.batch_size,
                epochs=args.epochs,
                validation_data=test_data,
                verbose=1
            )

    # Saving Logs
    logs = {
        'best_acc': max(fitting.history['val_accuracy']),
        'config': vars(args),
        'history': fitting.history,
    }
    print('Best score: ', logs['best_acc'])

    dir_path = os.path.dirname(os.path.realpath(__file__))
    logfile_name = dir_path + f'/logs/{args.model}-{int(time.time())}.json'
    os.makedirs(os.path.dirname(logfile_name), exist_ok=True)
    with open(logfile_name, 'w') as f:
        json.dump(logs, f, indent=4)
    print('Log file saved at: ', logfile_name)

def main(args):
    # Set random seeds
    np.random.seed(42)
    tf.random.set_seed(42)

    print("Version: ", tf.__version__)
    print("Eager mode: ", tf.executing_eagerly())
    print("GPU is", "available" if tf.config.list_physical_devices("GPU") else "NOT AVAILABLE")

    train_data, test_data = get_dataset(args.dataset, batch_size=args.batch_size)

    train(train_data, test_data)

if __name__ == '__main__':
    main(args)
