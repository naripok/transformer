'''
https://colab.research.google.com/github/tensorflow/examples/blob/master/community/en/transformer_chatbot.ipynb#scrollTo=rHMPkA2eQrpT
'''

from __future__ import absolute_import, division, print_function, unicode_literals

from .preprocessing import preprocess_sentence, load_conversations, \
        tokenize_and_filter

import tensorflow as tf
from tensorflow_datasets.core.features.text import SubwordTextEncoder
tf.random.set_seed(1234)

import os
import logging
import datetime
import pickle

logging.basicConfig(level=logging.INFO)

# tokenizer params
TARGET_VOCAB_SIZE = 2**14
MAX_SUBWORD_LENGTH = 20

# Training params
MAX_LENGTH = 32
BATCH_SIZE = 64
BUFFER_SIZE = 20000
EPOCHS = 100
WARMUP_STEPS = 4000
PATIENCE = 10
MIN_DELTA = 0.001
BASELINE = None

# Maximum number of samples to preprocess
MAX_SAMPLES = 50000

# Hyper-parameters
NUM_LAYERS = 2
D_MODEL = 256
NUM_HEADS = 8
UNITS = 512
DROPOUT = 0.1


model_dir = "./saved_model"
tokenizer_path = model_dir + '/saved_tokenizer.pickle'
log_dir = model_dir + '/logs/fit/' + datetime.datetime.now().strftime('%Y%m%d-%H%M%S')

if not os.path.exists(log_dir):
    os.makedirs(log_dir)


def scaled_dot_product_attention(query, key, value, mask):
    ''' Calculate the attention weights. '''

    matmul_qk = tf.matmul(query, key, transpose_b=True)

    # scale matmul_qk
    depth = tf.cast(tf.shape(key)[-1], tf.float32)
    logits = matmul_qk / tf.math.sqrt(depth)

    # add the mask to zero out padding tokens
    if mask is not None:
        logits += (mask * -1e9)

    # softmax is normalized on the last axis (seq_len_k)
    attention_weights = tf.nn.softmax(logits, axis=-1)

    output = tf.matmul(attention_weights, value)

    return output


class MultiHeadAttention(tf.keras.layers.Layer):
    def __init__(self, d_model, num_heads, name='multi_head_attention'):
        super(MultiHeadAttention, self).__init__(name=name)
        self.num_heads = num_heads
        self.d_model = d_model

        assert d_model % self.num_heads == 0

        self.depth = d_model // self.num_heads

        self.query_dense = tf.keras.layers.Dense(units=d_model)
        self.key_dense = tf.keras.layers.Dense(units=d_model)
        self.value_dense = tf.keras.layers.Dense(units=d_model)

        self.dense = tf.keras.layers.Dense(units=d_model)

    def split_heads(self, inputs, batch_size):
        inputs = tf.reshape(inputs, shape=(batch_size, -1, self.num_heads, self.depth))
        return tf.transpose(inputs, perm=[0, 2, 1, 3])

    def call(self, inputs):
        query, key, value, mask = inputs['query'], inputs['key'], \
        inputs['value'], inputs['mask']
        batch_size = tf.shape(query)[0]

        # linear layers
        query = self.query_dense(query)
        key = self.key_dense(key)
        value = self.value_dense(value)

        # split heads
        query = self.split_heads(query, batch_size)
        key = self.split_heads(key, batch_size)
        value = self.split_heads(value, batch_size)

        # scaled dot-product attention
        scaled_attention = scaled_dot_product_attention(query, key, value, mask)

        scaled_attention = tf.transpose(scaled_attention, perm=[0, 2, 1, 3])

        # concatenation of heads
        concat_attention = tf.reshape(scaled_attention, (batch_size, -1, self.d_model))

        # final linear layer
        outputs = self.dense(concat_attention)

        return outputs


def create_padding_mask(x):
    mask = tf.cast(tf.math.equal(x, 0), tf.float32)
    # (batch_size, 1, 1, sequence length)
    return mask[:, tf.newaxis, tf.newaxis, :]


def create_look_ahead_mask(x):
    seq_len = tf.shape(x)[1]
    look_ahead_mask = 1 - tf.linalg.band_part(tf.ones((seq_len, seq_len)), -1, 0)
    padding_mask = create_padding_mask(x)
    return tf.maximum(look_ahead_mask, padding_mask)


class PositionalEncoding(tf.keras.layers.Layer):
    def __init__(self, position, d_model):
        super(PositionalEncoding, self).__init__()
        self.pos_encoding = self.positional_encoding(position, d_model)

    def get_angles(self, position, i, d_model):
        angles = 1 / tf.pow(10000, (2 * (i // 2)) / tf.cast(d_model, tf.float32))
        return position * angles

    def positional_encoding(self, position, d_model):
        angle_rads = self.get_angles(
            position=tf.range(position, dtype=tf.float32)[:, tf.newaxis],
            i=tf.range(d_model, dtype=tf.float32)[tf.newaxis, :],
            d_model=d_model)
        # apply sin to even index in the array
        sines = tf.math.sin(angle_rads[:, 0::2])
        # apply cos to odd index in the array
        cosines = tf.math.cos(angle_rads[:, 1::2])

        pos_encoding = tf.concat([sines, cosines], axis=-1)
        pos_encoding = pos_encoding[tf.newaxis, ...]
        return tf.cast(pos_encoding, tf.float32)

    def call(self, inputs):
        return inputs + self.pos_encoding[:, :tf.shape(inputs)[1], :]


def encoder_layer(units, d_model, num_heads, dropout, name='encoder_layer'):
    inputs = tf.keras.Input(shape=(None, d_model), name='inputs')
    padding_mask = tf.keras.Input(shape=(1, 1, None), name='padding_mask')

    attention = MultiHeadAttention(
        d_model, num_heads, name='attention')({
            'query': inputs,
            'key': inputs,
            'value': inputs,
            'mask': padding_mask
        })
    attention = tf.keras.layers.Dropout(rate=dropout)(attention)
    attention = tf.keras.layers.LayerNormalization(epsilon=1e-6)(inputs + attention)

    outputs = tf.keras.layers.Dense(units=units, activation='relu')(attention)
    outputs = tf.keras.layers.Dense(units=d_model)(outputs)
    outputs = tf.keras.layers.Dropout(rate=dropout)(outputs)
    outputs = tf.keras.layers.LayerNormalization(epsilon=1e-6)(attention + outputs)

    return tf.keras.Model(inputs=[inputs, padding_mask], outputs=outputs, name=name)


def encoder(vocab_size,
            num_layers,
            units,
            d_model,
            num_heads,
            dropout,
            name='encoder'):
    inputs = tf.keras.Input(shape=(None,), name='inputs')
    padding_mask = tf.keras.Input(shape=(1, 1, None), name='padding_mask')

    embeddings = tf.keras.layers.Embedding(vocab_size, d_model)(inputs)
    embeddings *= tf.math.sqrt(tf.cast(d_model, tf.float32))
    embeddings = PositionalEncoding(vocab_size, d_model)(embeddings)

    outputs = tf.keras.layers.Dropout(rate=dropout)(embeddings)

    for i in range(num_layers):
        outputs = encoder_layer(
            units=units,
            d_model=d_model,
            num_heads=num_heads,
            dropout=dropout,
            name='encoder_layer_{}'.format(i),
        )([outputs, padding_mask])

    return tf.keras.Model(inputs=[inputs, padding_mask], outputs=outputs, name=name)


def decoder_layer(units, d_model, num_heads, dropout, name='decoder_layer'):
    inputs = tf.keras.Input(shape=(None, d_model), name='inputs')
    enc_outputs = tf.keras.Input(shape=(None, d_model), name='encoder_outputs')
    look_ahead_mask = tf.keras.Input(shape=(1, None, None), name='look_ahead_mask')
    padding_mask = tf.keras.Input(shape=(1, 1, None), name='padding_mask')

    attention1 = MultiHeadAttention(
        d_model, num_heads, name='attention_1')(inputs={
            'query': inputs,
            'key': inputs,
            'value': inputs,
            'mask': look_ahead_mask
        })
    attention1 = tf.keras.layers.LayerNormalization(
        epsilon=1e-6)(attention1 + inputs)

    attention2 = MultiHeadAttention(
        d_model, num_heads, name='attention_2')(inputs={
            'query': attention1,
            'key': enc_outputs,
            'value': enc_outputs,
            'mask': padding_mask
        })
    attention2 = tf.keras.layers.Dropout(rate=dropout)(attention2)
    attention2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)(attention2 + attention1)

    outputs = tf.keras.layers.Dense(units=units, activation='relu')(attention2)
    outputs = tf.keras.layers.Dense(units=d_model)(outputs)
    outputs = tf.keras.layers.Dropout(rate=dropout)(outputs)
    outputs = tf.keras.layers.LayerNormalization(epsilon=1e-6)(outputs + attention2)

    return tf.keras.Model(
        inputs=[inputs, enc_outputs, look_ahead_mask, padding_mask],
        outputs=outputs,
        name=name)


def decoder(vocab_size,
            num_layers,
            units,
            d_model,
            num_heads,
            dropout,
            name='decoder'):

    inputs = tf.keras.Input(shape=(None,), name='inputs')
    enc_outputs = tf.keras.Input(shape=(None, d_model), name='encoder_outputs')
    look_ahead_mask = tf.keras.Input(shape=(1, None, None), name='look_ahead_mask')
    padding_mask = tf.keras.Input(shape=(1, 1, None), name='padding_mask')

    embeddings = tf.keras.layers.Embedding(vocab_size, d_model)(inputs)
    embeddings *= tf.math.sqrt(tf.cast(d_model, tf.float32))
    embeddings = PositionalEncoding(vocab_size, d_model)(embeddings)

    outputs = tf.keras.layers.Dropout(rate=dropout)(embeddings)

    for i in range(num_layers):
        outputs = decoder_layer(
            units=units,
            d_model=d_model,
            num_heads=num_heads,
            dropout=dropout,
            name='decoder_layer_{}'.format(i),
        )(inputs=[outputs, enc_outputs, look_ahead_mask, padding_mask])

    return tf.keras.Model(
        inputs=[inputs, enc_outputs, look_ahead_mask, padding_mask],
        outputs=outputs,
        name=name)


def transformer(vocab_size,
                num_layers,
                units,
                d_model,
                num_heads,
                dropout,
                name='transformer'):

    inputs = tf.keras.Input(shape=(None,), name='inputs')
    dec_inputs = tf.keras.Input(shape=(None,), name='dec_inputs')

    enc_padding_mask = tf.keras.layers.Lambda(
        create_padding_mask, output_shape=(1, 1, None),
        name='enc_padding_mask')(inputs)

    # mask the future tokens for decoder inputs at the 1st attention block
    look_ahead_mask = tf.keras.layers.Lambda(
        create_look_ahead_mask,
        output_shape=(1, None, None),
        name='look_ahead_mask')(dec_inputs)

    # mask the encoder outputs for the 2nd attention block
    dec_padding_mask = tf.keras.layers.Lambda(
        create_padding_mask, output_shape=(1, 1, None),
        name='dec_padding_mask')(inputs)

    enc_outputs = encoder(
        vocab_size=vocab_size,
        num_layers=num_layers,
        units=units,
        d_model=d_model,
        num_heads=num_heads,
        dropout=dropout,
    )(inputs=[inputs, enc_padding_mask])

    dec_outputs = decoder(
        vocab_size=vocab_size,
        num_layers=num_layers,
        units=units,
        d_model=d_model,
        num_heads=num_heads,
        dropout=dropout,
    )(inputs=[dec_inputs, enc_outputs, look_ahead_mask, dec_padding_mask])

    outputs = tf.keras.layers.Dense(units=vocab_size, name='outputs')(dec_outputs)

    return tf.keras.Model(inputs=[inputs, dec_inputs], outputs=outputs, name=name)


def loss_function(y_true, y_pred):
    y_true = tf.reshape(y_true, shape=(-1, MAX_LENGTH - 1))

    loss = tf.keras.losses.SparseCategoricalCrossentropy(
        from_logits=True, reduction='none')(y_true, y_pred)

    mask = tf.cast(tf.not_equal(y_true, 0), tf.float32)
    loss = tf.multiply(loss, mask)

    return tf.reduce_mean(loss)


class CustomSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
    def __init__(self, d_model, warmup_steps=WARMUP_STEPS):
        super(CustomSchedule, self).__init__()
        self.d_model = d_model
        self.warmup_steps = warmup_steps
        self.d_model_t = tf.cast(self.d_model, tf.float32)

    def __call__(self, step):
        arg1 = tf.math.rsqrt(step)
        arg2 = step * (self.warmup_steps**-1.5)

        return tf.math.rsqrt(self.d_model_t) * tf.math.minimum(arg1, arg2)

    def get_config(self):
        return {
            'd_model': self.d_model,
            'warmup_steps': self.warmup_steps
        }


def accuracy(y_true, y_pred):
    # ensure labels have shape (batch_size, MAX_LENGTH - 1)
    y_true = tf.reshape(y_true, shape=(-1, MAX_LENGTH - 1))
    return tf.keras.metrics.sparse_categorical_accuracy(y_true, y_pred)


def make_tokenizer(data):
    logging.info('Training tokenizer...')

    tokenizer = SubwordTextEncoder.build_from_corpus(
            data,
            target_vocab_size=TARGET_VOCAB_SIZE,
            max_subword_length=MAX_SUBWORD_LENGTH
            )

    logging.info(f'Target Tokenizer vocab size: {TARGET_VOCAB_SIZE}')
    logging.info(f'Tokenizer vocab size: {tokenizer.vocab_size}')

    # save tokenizer
    logging.info('Saving tokenizer.')

    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    with open(tokenizer_path, 'wb+') as f:
        pickle.dump(tokenizer, f, protocol=pickle.HIGHEST_PROTOCOL)

    #  tokenizer.save_to_file(tokenizer_path)

    logging.info('Done!')

    return tokenizer


def make_dataset(inputs, outputs, tokenizer=None):
    if not tokenizer:
        tokenizer = make_tokenizer(inputs + outputs)

    inputs, outputs = tokenize_and_filter(tokenizer, inputs, outputs, MAX_LENGTH)

    logging.info('Making data set.')
    # decoder inputs use the previous target as input
    # remove START_TOKEN from targets
    dataset = tf.data.Dataset.from_tensor_slices((
        {
            'inputs': inputs,
            'dec_inputs': outputs[:, :-1]
        },
        {
            'outputs': outputs[:, 1:]
        },
    ))

    dataset = dataset.cache() \
                    .shuffle(BUFFER_SIZE) \
                    .batch(BATCH_SIZE) \
                    .prefetch(tf.data.experimental.AUTOTUNE)

    return tokenizer, dataset


def make_model(tokenizer):
    logging.info('Compiling model.')

    learning_rate = CustomSchedule(D_MODEL)

    optimizer = tf.keras.optimizers.Adam(
        learning_rate, beta_1=0.9, beta_2=0.98, epsilon=1e-9)

    # Vocabulary size plus start and end token
    vocab_size = tokenizer.vocab_size + 2

    model = transformer(
        vocab_size=vocab_size,
        num_layers=NUM_LAYERS,
        units=UNITS,
        d_model=D_MODEL,
        num_heads=NUM_HEADS,
        dropout=DROPOUT)

    model.compile(optimizer=optimizer, loss=loss_function, metrics=[accuracy])

    logging.info('Compiled model.')
    logging.info(model.summary())

    return tokenizer, model


def train(tokenizer, model, data, epochs=EPOCHS, min_delta=MIN_DELTA, patience=PATIENCE, baseline=BASELINE):
    # reset session
    tf.keras.backend.clear_session()

    if not model:
        tokenizer, model = make_model(tokenizer)

    # training callbacks
    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor='loss', min_delta=min_delta, patience=patience,
        verbose=1, mode='auto', baseline=baseline,
        restore_best_weights=True
        )
    tensor_board = tf.keras.callbacks.TensorBoard(
        log_dir=log_dir, histogram_freq=0, write_graph=False, write_images=True,
        update_freq=50, profile_batch=2, embeddings_freq=10,
        embeddings_metadata=None
        )
    terminate_on_nan = tf.keras.callbacks.TerminateOnNaN()

    # Create a callback that saves the model's weights
    logging.info('Training model.')
    try:
        model.fit(
                data,
                epochs=epochs,
                callbacks=[
                    early_stopping,
                    tensor_board,
                    terminate_on_nan
                    ]
                )

    except KeyboardInterrupt:
        logging.info('\nTraining Interruped!')

    finally:
        logging.info('Saving model.')
        model.save(model_dir)

    return tokenizer, model


def evaluate(tokenizer, model, sentence, training=False):
    START_TOKEN, END_TOKEN = [tokenizer.vocab_size], [tokenizer.vocab_size + 1]

    sentence = preprocess_sentence(sentence)
    sentence = tf.expand_dims(START_TOKEN + tokenizer.encode(sentence) + END_TOKEN, axis=0)
    logging.info(sentence)
    output = tf.expand_dims(START_TOKEN, 0)

    for i in range(MAX_LENGTH):
        predictions = model(inputs=[sentence, output], training=training)

        # select the last word from the seq_len dimension
        predictions = predictions[:, -1:, :]
        predicted_id = tf.cast(tf.argmax(predictions, axis=-1), tf.int32)

        # return the result if the predicted_id is equal to the end token
        if tf.equal(predicted_id, END_TOKEN[0]):
            break

        # concatenated the predicted_id to the output which is given to the decoder
        # as its input.
        output = tf.concat([output, predicted_id], axis=-1)

    return tf.squeeze(output, axis=0)


def predict(tokenizer, model, sentence, training=False):
    prediction = evaluate(tokenizer, model, sentence, training).numpy()
    logging.info(prediction)
    prediction = [i for i in prediction if i < tokenizer.vocab_size]
    return tokenizer.decode(prediction)


def learn_step(in_data, out_data, model):
    pass


def load_model():
    logging.info('Loading tokenizer.')

    with open(tokenizer_path, 'rb') as f:
        tokenizer = pickle.load(f)
    #  tokenizer = SubwordTextEncoder.load_from_file(tokenizer_path)

    logging.info(f'Tokenizer vocab size: {tokenizer.vocab_size}.')

    logging.info('Loading model.')
    model = tf.keras.models.load_model(
            model_dir,
            custom_objects={
                'CustomSchedule': CustomSchedule,
                'loss_function': loss_function
                }
            )

    logging.info('Loaded model.')
    logging.info(model.summary())

    logging.info('Done!')
    return tokenizer, model


if __name__ == '__main__':

    NEW_MODEL = False
    TRAIN = True

    questions, answers = load_conversations(MAX_SAMPLES)

    if NEW_MODEL:
        tokenizer, data = make_dataset(questions, answers)
        tokenizer, model = make_model(tokenizer)

        if TRAIN:
            tokenizer, model = train(tokenizer, model, data)
    else:
        tokenizer, model = load_model()

        if TRAIN:
            tokenizer, data = make_dataset(questions, answers, tokenizer)
            tokenizer, model = train(tokenizer, model, data)

    # feed the model with its previous output
    sentence = 'I am not crazy, my mother had me tested.'
    print(sentence)
    for _ in range(10):
        sentence = predict(tokenizer, model, sentence)
        print(sentence)
