import tensorflow as tf
from tensorflow.keras import layers
from datasets import DatasetWrapper
from models import get_model
from trainers.Trainer import Trainer

from utils.conlleval import evaluate

class CalculateMetrics(tf.keras.callbacks.Callback):
    def __init__(self, dataset, id_to_label) -> None:
        self.id_to_label = list(id_to_label.values())
        self.history = []
        self.int_to_tag = tf.keras.layers.StringLookup(vocabulary=self.id_to_label, invert=True)

        self.dataset = dataset
        self.y_true = [y for y in self.dataset.map(lambda x, y: y)]
        self.y_true = tf.reshape(self.y_true, [-1, self.dataset.element_spec[1].shape[-1]])
        self.y_true_tag = self.int_to_tag(tf.argmax(self.y_true, axis=-1) + 1).numpy().astype(dtype='<U13')

    def on_epoch_end(self, epoch, logs={}):
        out = self.model.predict(self.dataset)
        y_pred = tf.reshape(out, self.y_true.shape)
        y_pred_tag = self.int_to_tag(tf.argmax(y_pred, axis=-1) + 1).numpy().astype(dtype='<U13')

        _, _, f1_score = evaluate(self.y_true_tag, y_pred_tag)
        logs['val_f1_score'] = f1_score
        self.history.append(f1_score)

def train(args, dataset: DatasetWrapper):
    vectorize_layer = layers.TextVectorization(
        output_mode='int',
    )

    train_data, test_data, ds_info = dataset.getData(args.batch_size)
    if not 'id_to_label' in ds_info.metadata \
        or not type(ds_info.metadata['id_to_label']) is dict \
        or ds_info.metadata['id_to_label']['0'] != 'O':
        raise "dataset must have `id_to_label`: dict(int, tag_str) for NERTrainer and tag `O` must be '0'"

    train_text = train_data.flat_map(lambda text, label: tf.data.Dataset.from_tensor_slices(text))
    vectorize_layer.adapt(train_text)
    vocab_size = vectorize_layer.vocabulary_size()
    print('Vocab size:', vocab_size)

    train_data_size = len(train_data)
    truncated_size = (*train_data.as_numpy_iterator().next()[0].shape[:-1], args.seq_len)

    # some unknown `vectorize_layer` issus cause `x.shape` inconsistently
    train_data = train_data.shuffle(1000)\
                           .map(lambda x, y: (tf.squeeze(vectorize_layer(tf.expand_dims(x, axis=-1))), y))\
                           .filter(lambda x, y: tf.py_function(lambda x: x.shape == truncated_size, [x], tf.bool))

    test_data = test_data.shuffle(1000)\
                         .map(lambda x, y: (tf.squeeze(vectorize_layer(tf.expand_dims(x, axis=-1))), y))\
                         .filter(lambda x, y: tf.py_function(lambda x: x.shape == truncated_size, [x], tf.bool))

    model = tf.keras.models.Sequential([
        tf.keras.layers.Input(shape=(args.seq_len,), dtype=tf.int64),
        get_model(args, vocab_size),
        tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(dataset.getOutputSize())),
    ])

    lr_decayed_fn = tf.keras.optimizers.schedules.CosineDecay(args.lr, args.epochs * args.steps_per_epoch, alpha=1e-2)
    trainer = Trainer(
        args,
        model,
        monitor='val_f1_score',
        optimizer=tf.keras.optimizers.Adam(learning_rate=lr_decayed_fn, beta_1=0.9, beta_2=0.98, epsilon=1e-07),
        loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True),
        metrics=['accuracy']
    )

    return trainer.train(train_data, test_data, callbacks=[CalculateMetrics(test_data, ds_info.metadata['id_to_label'])])
