import tensorflow as tf
from tensorflow.keras import layers
from datasets import DatasetWrapper, MSRADataset
from models import get_model
from .Trainer import Trainer
from utils.conlleval import evaluate
from utils.lr_finder import LRFinder, MaxStepStoppingWithLogging

def calculate_metrics(model, dataset):
    all_true_tag_ids, all_predicted_tag_ids = [], []

    for x, y in dataset:
        output = model.predict(x)
        predictions = tf.argmax(output, axis=-1)
        predictions = tf.reshape(predictions, [-1])

        ground_truth = tf.argmax(y, axis=-1)
        true_tag_ids = tf.reshape(ground_truth, [-1])

        all_true_tag_ids.append(true_tag_ids)
        all_predicted_tag_ids.append(predictions)

    all_true_tag_ids = tf.concat(all_true_tag_ids, axis=-1)
    all_predicted_tag_ids = tf.concat(all_predicted_tag_ids, axis=-1)

    predicted_tags = [MSRADataset.ID_TO_LABEL[int(tag)] for tag in all_predicted_tag_ids]
    real_tags = [MSRADataset.ID_TO_LABEL[int(tag)] for tag in all_true_tag_ids]

    try:
        evaluate(real_tags, predicted_tags)
    except ZeroDivisionError:
        print('The accuracy in model log is for `O` tag. Real accuracy = 0.')
    except:
        print('Unkown error')

def train(args, dataset: DatasetWrapper):
    vectorize_layer = layers.TextVectorization(
        output_mode='int',
    )

    train_data, test_data = dataset.getData(args.batch_size)
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

    lr_decayed_fn = tf.keras.optimizers.schedules.CosineDecay(args.lr, args.epochs * train_data_size, alpha=1e-2)
    trainer = Trainer(
        args,
        model,
        optimizer=tf.keras.optimizers.Adam(learning_rate=lr_decayed_fn, beta_1=0.9, beta_2=0.98, epsilon=1e-07),
        loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True),
        metrics=['accuracy']
    )

    return trainer.train(train_data, test_data)
