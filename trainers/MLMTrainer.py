import tensorflow as tf
from tensorflow.keras import layers
from datasets import DatasetWrapper
from models import get_model, MLM
from transformers import AutoTokenizer
from utils.mlm_utils import get_masked_input_and_labels

def train(args, dataset: DatasetWrapper):
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

    return fitting