import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
from tensorflow.keras import layers
from tensorflow.keras import losses
from model import QuanformerEncoder

# tf.get_logger().setLevel('ERROR')
tf.keras.backend.set_floatx('float64')

# Set random seeds
np.random.seed(42)
tf.random.set_seed(42)

print("Version: ", tf.__version__)
print("Eager mode: ", tf.executing_eagerly())
print("GPU is", "available" if tf.config.list_physical_devices("GPU") else "NOT AVAILABLE")

train_raw, test_raw = tfds.load(
    name="imdb_reviews",
    split=('train', 'test'),
    as_supervised=True
)

vectorize_layer = layers.TextVectorization(
    standardize="lower_and_strip_punctuation",
    output_mode='int',
    output_sequence_length=8
)

train_text = tf.data.Dataset.from_tensor_slices([ text for text, label in train_raw ])
vectorize_layer.adapt(train_text)
max_features = vectorize_layer.vocabulary_size()
print('Vocab size:', max_features)

model = tf.keras.models.Sequential([
    tf.keras.Input(shape=(1,), dtype=tf.string),
    vectorize_layer,
    layers.Embedding(max_features + 1, 2 * 3),
    layers.Flatten(),
    QuanformerEncoder(
        embed_size=2,
        src_seq_len=8,
        num_blocks=1,
    ),
    layers.Dense(2, 'softmax'),
])

opt = tf.keras.optimizers.Adam(learning_rate=0.0001, beta_1=0.9, beta_2=0.98, epsilon=1e-9)
model.compile(
    opt,
    loss=tf.keras.losses.CategoricalCrossentropy(),
    metrics=["accuracy"]
)

print(model.summary())
train_data = tf.data.Dataset.from_tensor_slices((
                    [ text for text, label in train_raw ],
                    [ tf.one_hot(label, 2) for text, label in train_raw ],
))
test_data = tf.data.Dataset.from_tensor_slices((
                    [ text for text, label in test_raw ],
                    [ tf.one_hot(label, 2) for text, label in test_raw ],
))

fitting = model.fit(
            train_data.shuffle(1000).batch(16),
            batch_size=16,
            epochs=10,
            validation_data=test_data.batch(16),
            verbose=1
        )
