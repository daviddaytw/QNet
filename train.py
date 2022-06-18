import argparse, os, uuid, json
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
from datasets import get_dataset, get_dataset_output_size
from models import get_model

tf.keras.backend.set_floatx('float64')

parser = argparse.ArgumentParser(description='Configure training arugments.', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--dataset', '-d', default='stackoverflow', type=str,
                    help='Select the training dataset.')
parser.add_argument('--model', '-m', default='transformer', type=str,
                    help='Select the trainig model (transformer, qnet, fnet)')
parser.add_argument('--seq_len', '-ml', default='8', type=int,
                    help='Input length for the model.')
parser.add_argument('--embed_size', '-ed', default='2', type=int,
                    help='Embedding size for each token.')
parser.add_argument('--num_blocks', '-nb', default='1', type=int,
                    help='Number of mini-blocks in the model.')
parser.add_argument('--batch_size', '-bs', default='128', type=int,
                    help='Number of samples per batch.')
parser.add_argument('--lr', '-lr', default='0.01', type=float,
                    help='The initial learning rate.')
parser.add_argument('--epochs','-e', default='10', type=int,
                    help='Number of training loops over all training data')
args = parser.parse_args()

# Set random seeds
np.random.seed(42)
tf.random.set_seed(42)

print("Version: ", tf.__version__)
print("Eager mode: ", tf.executing_eagerly())
print("GPU is", "available" if tf.config.list_physical_devices("GPU") else "NOT AVAILABLE")

train_data, test_data = get_dataset(args.dataset)

vectorize_layer = layers.TextVectorization(
    standardize="lower_and_strip_punctuation",
    output_mode='int',
    output_sequence_length=args.seq_len
)

train_text = tf.data.Dataset.from_tensor_slices([ text for text, label in train_data ])
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
    layers.Dense(get_dataset_output_size(args.dataset)),
])

lr_decayed_fn = tf.keras.optimizers.schedules.CosineDecay(args.lr, 16, alpha=1e-6)
opt = tf.keras.optimizers.Adam(learning_rate=lr_decayed_fn,)
model.compile(
    opt,
    loss=tf.keras.losses.CategoricalCrossentropy(),
    metrics=["accuracy"]
)

print(model.summary())

fitting = model.fit(
            train_data.shuffle(1000).batch(args.batch_size),
            batch_size=args.batch_size,
            epochs=args.epochs,
            validation_data=test_data.batch(args.batch_size),
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
logfile_name = dir_path + '/logs/' + str(uuid.uuid4()) + '.json'
os.makedirs(os.path.dirname(logfile_name), exist_ok=True)
with open(logfile_name, 'w') as f:
    json.dump(logs, f, indent=4)
print('Log file saved at: ', logfile_name)
