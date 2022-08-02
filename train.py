# since we encounter this [problem](https://colab.research.google.com/github/tensorflow/docs/blob/master/site/en/tutorials/distribute/multi_worker_with_keras.ipynb?hl=id-IDCache#scrollTo=Mhq3fzyR5hTw)
# it must put on the file starts before initialize other class and after set env.
from utils.args_parser import solve_args
args = solve_args(multi_worker_strategy=True)

import os, time, json

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
from datasets import DatasetWrapper, get_dataset
from models import get_model, MLM
from transformers import AutoTokenizer
from utils.mlm_utils import get_masked_input_and_labels

def mlm_evaluation(dataset: DatasetWrapper):
    all_data = dataset.getData(args.batch_size)
    tokenizer = AutoTokenizer.from_pretrained('AIRI-Institute/gena-lm-bert-base')
    mask_token_id = tokenizer.convert_tokens_to_ids("[MASK]")

    def encode(inputs):
        texts = tokenizer(
                tf.compat.as_str(inputs.numpy()),
                max_length=args.seq_len,
                add_special_tokens=True,
                return_tensors='np',
                return_token_type_ids=False,
                return_attention_mask=False)['input_ids']
        texts = get_masked_input_and_labels(texts.squeeze(), mask_token_id)
        return texts

    def tf_encode(inputs):
        result = tf.py_function(func=encode, inp=[inputs['text']], Tout=[tf.int64, tf.int64, tf.int64])
        return result

    all_data = all_data.map(tf_encode).batch(args.batch_size)

    vocab_size = len(tokenizer)

    inputs = tf.keras.Input(shape=(args.seq_len,), dtype=tf.int64)
    outputs = get_model(args, vocab_size)(inputs)
    outputs = layers.Dense(vocab_size, activation="softmax")(outputs)

    model = MLM.MaskedLanguageModel(inputs, outputs)

    lr_decayed_fn = tf.keras.optimizers.schedules.CosineDecay(args.lr, args.epochs * len(all_data), alpha=1e-2)
    opt = tf.keras.optimizers.Adam(learning_rate=lr_decayed_fn, beta_1=0.9, beta_2=0.98, epsilon=1e-09)
    model.compile(opt)

    print(model.summary())
    fitting = model.fit(
                all_data,
                batch_size=args.batch_size,
                epochs=args.epochs,
                verbose=1
            )

    save_log(fitting.history)

def train(dataset: DatasetWrapper):
    vectorize_layer = layers.TextVectorization(
        standardize="lower_and_strip_punctuation",
        output_mode='int',
        output_sequence_length=args.seq_len
    )

    train_data, test_data = dataset.getData(args.batch_size)

    train_text = train_data.flat_map(lambda text, label: tf.data.Dataset.from_tensor_slices(text))
    vectorize_layer.adapt(train_text)
    vocab_size = vectorize_layer.vocabulary_size()
    print('Vocab size:', vocab_size)

    model = tf.keras.models.Sequential([
        tf.keras.Input(shape=(1,), dtype=tf.string),
        vectorize_layer,
        get_model(args, vocab_size),
        layers.GlobalAveragePooling1D(),
        layers.Dense(dataset.getOutputSize()),
    ])

    lr_decayed_fn = tf.keras.optimizers.schedules.CosineDecay(args.lr, args.epochs * len(train_data), alpha=1e-2)
    opt = tf.keras.optimizers.Adam(learning_rate=lr_decayed_fn, beta_1=0.9, beta_2=0.98, epsilon=1e-09)

    if dataset.getOutputSize() > 2:
        model.compile(
            opt,
            loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True),
            metrics=['categorical_accuracy']
        )
    else:
        model.compile(
            opt,
            loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
            metrics=['binary_accuracy']
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
    if dataset.getOutputSize() > 2:
        save_log(fitting.history, 'val_categorical_accuracy')
    else:
        save_log(fitting.history, 'val_binary_accuracy')

def save_log(history, val_metric: str=None):
    logs = {
        'config': vars(args),
        'history': history,
    }

    if val_metric != None:
        logs['best_acc'] = max(history[val_metric])
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

    dataset = get_dataset(args.dataset)
    if dataset.getTask() == 'classification':
        train(dataset)
    if dataset.getTask() == 'mlm':
        mlm_evaluation(dataset)

if __name__ == '__main__':
    main(args)
