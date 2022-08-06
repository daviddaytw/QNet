import tensorflow as tf
import tensorflow_datasets as tfds
from tensorflow.keras import layers
from datasets import DatasetWrapper
from models import get_model

def encode(args, inputs, vectorize_layer):
    texts = tf.strings.bytes_split(inputs)
    encoded_texts = tf.squeeze(vectorize_layer(tf.expand_dims(texts, axis=-1)))
    # hard code (0, 0), (args.batch_size, args.seq_len) in this. looking for better way
    encoded_texts = tf.slice(encoded_texts, (0, 0), (args.batch_size, args.seq_len))

    return encoded_texts

def pretrain(args, vectorize_layer):
    vocab_size = vectorize_layer.vocabulary_size()
    mask_token_id = vectorize_layer(['[MASK]']).numpy()[0][0]
    def random_masked(inputs, mask_ratio=0.15):
        encoded_texts = encode(args, inputs['text'], vectorize_layer)

        # 15% masking
        inp_mask = tf.random.uniform([*encoded_texts.shape]) < mask_ratio
        # Do not mask special tokens
        inp_mask = tf.where(encoded_texts < 2, False, inp_mask)
        # Prepare input
        encoded_texts_masked = tf.where(inp_mask, mask_token_id, encoded_texts)

        # y_labels would be same as encoded_texts i.e input tokens
        y_labels = encoded_texts

        # Prepare sample_weights to pass to .fit() method
        sample_weights = tf.where(inp_mask, 0, 1)

        return encoded_texts_masked, y_labels, sample_weights

    all_data = tfds.load('T2TDataset', split='train')\
                      .shuffle(1000)\
                      .batch(args.batch_size, drop_remainder=True)\
                      .map(random_masked)

    model = tf.keras.models.Sequential([
        tf.keras.Input(shape=(args.seq_len,), dtype=tf.int64),
        get_model(args, vocab_size),
        layers.Dense(vocab_size),
    ])

    lr_decayed_fn = tf.keras.optimizers.schedules.CosineDecay(args.lr, args.epochs * len(all_data), alpha=1e-2)
    opt = tf.keras.optimizers.Adam(learning_rate=lr_decayed_fn, beta_1=0.9, beta_2=0.98, epsilon=1e-09)
    model.compile(opt, loss='sparse_categorical_crossentropy', loss='sparse_categorical_crossentropy')

    if args.model_path.with_suffix(".ckpt.index").exists():
        print('Load pretrained weight: {}'.format(args.model_path.with_suffix(".ckpt")))
        model.load_weights(args.model_path.with_suffix(".ckpt"))
    else:
        print(model.summary())
        print('Training pretrain Mask LM')
        model.fit(
            all_data,
            batch_size=args.batch_size,
            epochs=args.epochs,
            verbose=1
        )

        model.save_weights(args.model_path.with_suffix(".ckpt"))

    return model

def train(args, dataset: DatasetWrapper):
    if args.model_path is None:
        raise ValueError("MLM task must specify model path")

    vectorize_layer = layers.TextVectorization(
        output_mode='int',
    )

    vectorize_layer.adapt(['a', 't', 'c', 'g', '[MASK]'])
    vocab_size = vectorize_layer.vocabulary_size()
    print('Vocab size:', vocab_size)

    train_data, test_data = dataset.getData(args.batch_size)
    train_data = train_data.map(lambda x, y: (encode(args, x, vectorize_layer), y))
    test_data = test_data.map(lambda x, y: (encode(args, x, vectorize_layer), y))

    model = pretrain(args, vectorize_layer)
    model.trainable = False
    outputs = model.layers[-1].output
    outputs = layers.GlobalMaxPooling1D()(outputs)
    outputs = layers.Dense(64, activation="relu")(outputs)
    outputs = layers.Dense(dataset.getOutputSize())(outputs)
    end2end_model = tf.keras.Model(model.input, outputs)

    lr_decayed_fn = tf.keras.optimizers.schedules.CosineDecay(args.lr, args.epochs * len(train_data), alpha=1e-2)
    opt = tf.keras.optimizers.Adam(learning_rate=lr_decayed_fn, beta_1=0.9, beta_2=0.98, epsilon=1e-09)

    if dataset.getOutputSize() > 2:
        end2end_model.compile(
            opt,
            loss=tf.keras.losses.CategoricalCrossentropy(),
            metrics=['categorical_accuracy']
        )
    else:
        end2end_model.compile(
            opt,
            loss=tf.keras.losses.BinaryCrossentropy(),
            metrics=['binary_accuracy']
        )

    print(end2end_model.summary())

    fitting = end2end_model.fit(
                train_data,
                batch_size=args.batch_size,
                epochs=args.epochs,
                validation_data=test_data,
                validation_data=test_data,
                verbose=1
            )

    return fitting