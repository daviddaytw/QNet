import tensorflow as tf
from tensorflow.keras import layers
from datasets import DatasetWrapper
from models import get_model
from trainers.Trainer import Trainer

def train(args, dataset: DatasetWrapper):
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
    opt = tf.keras.optimizers.Adam(learning_rate=lr_decayed_fn, beta_1=0.9, beta_2=0.98, epsilon=1e-07)

    if dataset.getOutputSize() > 2:
        trainer = Trainer(args, model,
            optimizer=opt,
            loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True),
            metrics=['categorical_accuracy'],
        )
    else:
        trainer = Trainer(args, model,
            optimizer=opt,
            loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
            metrics=['binary_accuracy']
        )

    return trainer.train(train_data, test_data)
