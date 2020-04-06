# -*- coding: utf-8 -*-
"""context_twin_transformer.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/11lsLAg5GsPLRY5d2287_IEioRJEMeccj

Reference:
  - [Colab Transformer Model Chatbot](https://colab.research.google.com/github/tensorflow/examples/blob/master/community/en/transformer_chatbot.ipynb#scrollTo=rHMPkA2eQrpT)
  - [Transformer-XL paper](https://arxiv.org/pdf/1901.02860.pdf)
  - [Attention Is All You Need](https://arxiv.org/pdf/1706.03762.pdf)
  - [Convokit docs](https://convokit.cornell.edu/)
  - [Tensorflow v2.2.0rc2 API docs](https://www.tensorflow.org/versions/r2.2/api_docs/python/tf)
"""

#@title Transformer Network Chatbot Training Routine { run: "auto", display-mode: "form" }

from __future__ import absolute_import, division, print_function, unicode_literals

IS_COLAB = True #@param {type:"boolean"}
MOUNT_DRIVE = True #@param {type:"boolean"}

if IS_COLAB and MOUNT_DRIVE:
    from google.colab import drive
    drive.mount('/content/drive', force_remount=True)

# Commented out IPython magic to ensure Python compatibility.
#@title Environment Setup {display-mode: "form"}

# This code will be hidden when the notebook is loaded.

# %tensorflow_version 2.x

import re
import logging
import pickle
from tqdm import tqdm
import tensorflow as tf
from tensorflow_datasets.core.features.text import SubwordTextEncoder

tf.random.set_seed(42)
logging.basicConfig(level=logging.INFO)

tf.keras.backend.clear_session()

IS_TPU = False
if IS_COLAB:
    from google.colab import output
    try:
        with output.use_tags('setup'):
            !pip install convokit
            !python3 -m spacy download en
            tpu = tf.distribute.cluster_resolver.TPUClusterResolver()  # TPU detection
            print('Running ovariable_namen TPU ', tpu.cluster_spec().as_dict()['worker'])
            tf.config.experimental_connect_to_cluster(tpu)
            tf.tpu.experimental.initialize_tpu_system(tpu)
            tpu_strategy = tf.distribute.experimental.TPUStrategy(tpu)
            IS_TPU = True
        output.clear(output_tags='setup')
    
    except ValueError:
        logging.info('Not connected to a TPU runtime')

logging.info('Done!')

#@title Save path {display-mode: "form"}

# This code will be hidden when the notebook is loaded.

import os
import datetime

if IS_COLAB:
    model_path = "/content/drive/My Drive/discordbot/saved_model"  #@param {type:"string"}
else:
    model_path = "./saved_model"  #@param {type:"string"}

model_weights_path = model_path + '/weights.h5'
tokenizer_path = model_path + '/saved_tokenizer.pickle'
model_config_path = model_path + '/model_config.pickle'
dataset_config_path = model_path + '/dataset_config.pickle'
train_config_path = model_path + '/train_config.pickle'
log_dir = model_path + '/logs/fit/' + datetime.datetime.now().strftime('%Y%m%d-%H%M%S')

# if not os.path.exists(log_dir):
#     os.makedirs(log_dir)

#@title Preprocessing Functions {display-mode: "form"}

import pandas as pd
import altair as alt
import convokit
from convokit import Corpus, download
import nltk; nltk.download('punkt')

# This code will be hidden when the notebook is loaded.

def tokenize_and_filter(tokenizer, inputs, context, outputs, max_length=32):
    # Define start and end token to indicate the start and end of a sentence
    START_TOKEN, END_TOKEN = [tokenizer.vocab_size], [tokenizer.vocab_size + 1]

    logging.info('Tokenizing sentences.')

    tokenized_inputs, tokenized_context, tokenized_outputs = [], [], []
    for (sentence1, sentence2, sentence3) in tqdm(zip(inputs, context, outputs)):
        # tokenize sentence
        sentence1 = START_TOKEN + tokenizer.encode(sentence1) + END_TOKEN
        sentence2 = START_TOKEN + tokenizer.encode(sentence2) + END_TOKEN
        sentence3 = START_TOKEN + tokenizer.encode(sentence3) + END_TOKEN

        # check tokenized sentence max length
        if len(sentence1) <= max_length \
        and len(sentence2) <= max_length \
        and len(sentence3) <= max_length:
            tokenized_inputs.append(sentence1)
            tokenized_context.append(sentence2)
            tokenized_outputs.append(sentence3)

    # pad tokenized sentences
    tokenized_inputs = tf.keras.preprocessing.sequence.pad_sequences(
        tokenized_inputs, maxlen=max_length, padding='post')
    tokenized_context = tf.keras.preprocessing.sequence.pad_sequences(
        tokenized_context, maxlen=max_length, padding='post')
    tokenized_outputs = tf.keras.preprocessing.sequence.pad_sequences(
        tokenized_outputs, maxlen=max_length, padding='post')

    logging.info('Done!')

    return tokenized_inputs, tokenized_context, tokenized_outputs


def preprocess_sentence(sentence):
    sentence = sentence.lower().strip()
    # creating a space between a word and the punctuation following it
    # eg: "he is a boy." => "he is a boy ."
    sentence = re.sub(r'([?.!,])', r' \1 ', sentence)
    sentence = re.sub(r'[" "]+', ' ', sentence)
    # replacing everything with space except (a-z, A-Z, ".", "?", "!", ",")
    sentence = re.sub(r'[^a-zA-Z?.!,]+', ' ', sentence)
    sentence = sentence.strip()
    # adding a start and an end token to the sentence
    return sentence


def load_conversations(corpus_name, max_samples, eval_percent=0.1):
    logging.info('Loading data.')

    def split_data(inputs, context, outputs, eval_percent):
        eval_index = int(len(inputs) * (1 - eval_percent))
        return (inputs[:eval_index],
                context[:eval_index],
                outputs[:eval_index],
                inputs[eval_index:],
                context[eval_index:],
                outputs[eval_index:])

    corpus = Corpus(filename=download(corpus_name))

    deleted_filter = re.compile(r'^(\[deleted]|\[removed])$')
    
    inputs, context, outputs = [], [], []
    for paths in corpus.iter_conversations():
        for path in paths.get_root_to_leaf_paths():
            for i in range(1, len(path)-1):

                if deleted_filter.match(path[i].text) \
                or deleted_filter.match(path[i-1].text) \
                or deleted_filter.match(path[i+1].text):
                    continue
                
                inputs.append(path[i].text)
                context.append(path[i-1].text)
                outputs.append(path[i+1].text)

                if len(inputs) >= max_samples:
                    return split_data(inputs, context, outputs, eval_percent)

    return split_data(inputs, context, outputs, eval_percent)

#@title Model Definition Functions {display-mode: "form"}

# This code will be hidden when the notebook is loaded.


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
    input_enc_outputs = tf.keras.Input(shape=(None, d_model), name='input_encoder_outputs')
    context_enc_outputs = tf.keras.Input(shape=(None, d_model), name='context_encoder_outputs')

    look_ahead_mask = tf.keras.Input(shape=(1, None, None), name='look_ahead_mask')
    input_padding_mask = tf.keras.Input(shape=(1, 1, None), name='input_padding_mask')
    context_padding_mask = tf.keras.Input(shape=(1, 1, None), name='context_padding_mask')

    attention1 = MultiHeadAttention(
        d_model, num_heads, name='attention_1')(inputs={
            'query': inputs,
            'key': inputs,
            'value': inputs,
            'mask': look_ahead_mask
        })
    attention1 = tf.keras.layers.Dropout(rate=dropout)(attention1)
    attention1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)(attention1)

    attention2 = MultiHeadAttention(
        d_model, num_heads, name='attention_2')(inputs={
            'query': attention1,
            'key': input_enc_outputs,
            'value': input_enc_outputs,
            'mask': input_padding_mask
        })
    attention2 = tf.keras.layers.Dropout(rate=dropout)(attention2)
    attention2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)(attention2 + attention1)

    attention3 = MultiHeadAttention(
        d_model, num_heads, name='attention_3')(inputs={
            'query': attention1,
            'key': context_enc_outputs,
            'value': context_enc_outputs,
            'mask': context_padding_mask
        })
    attention3 = tf.keras.layers.Dropout(rate=dropout)(attention3)
    attention3 = tf.keras.layers.LayerNormalization(epsilon=1e-6)(attention3 + attention1)

    outputs = tf.keras.layers.Dense(units=units, activation='relu')(tf.concat(
        [attention1, attention2, attention3], axis=2, name='attention_concat'))
    outputs = tf.keras.layers.Dropout(rate=dropout)(outputs)
    outputs = tf.keras.layers.LayerNormalization(epsilon=1e-6)(outputs)

    outputs = tf.keras.layers.Dense(units=d_model)(outputs)
    outputs = tf.keras.layers.Dropout(rate=dropout)(outputs)
    outputs = tf.keras.layers.LayerNormalization(epsilon=1e-6)(outputs)

    return tf.keras.Model(
        inputs=[inputs,
                input_enc_outputs,
                context_enc_outputs,
                look_ahead_mask,
                input_padding_mask,
                context_padding_mask
                ],
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
    input_enc_outputs = tf.keras.Input(shape=(None, d_model), name='input_encoder_outputs')
    context_enc_outputs = tf.keras.Input(shape=(None, d_model), name='context_encoder_outputs')

    look_ahead_mask = tf.keras.Input(shape=(1, None, None), name='look_ahead_mask')
    input_enc_padding_mask = tf.keras.Input(shape=(1, 1, None), name='input_end_padding_mask')
    context_enc_padding_mask = tf.keras.Input(shape=(1, 1, None), name='context_end_padding_mask')

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
        )(inputs=[
                  outputs,
                  input_enc_outputs,
                  context_enc_outputs,
                  look_ahead_mask,
                  input_enc_padding_mask,
                  context_enc_padding_mask
                  ])

    return tf.keras.Model(
        inputs=[inputs,
                input_enc_outputs,
                context_enc_outputs,
                look_ahead_mask,
                input_enc_padding_mask,
                context_enc_padding_mask
                ],
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
    context = tf.keras.Input(shape=(None,), name='context')
    dec_inputs = tf.keras.Input(shape=(None,), name='dec_inputs')

    input_enc_padding_mask = tf.keras.layers.Lambda(
        create_padding_mask, output_shape=(1, 1, None),
        name='input_enc_padding_mask')(inputs)

    context_enc_padding_mask = tf.keras.layers.Lambda(
        create_padding_mask, output_shape=(1, 1, None),
        name='context_enc_padding_mask')(context)

    # mask the future tokens for decoder inputs at the 1st attention block
    look_ahead_mask = tf.keras.layers.Lambda(
        create_look_ahead_mask,
        output_shape=(1, None, None),
        name='look_ahead_mask')(dec_inputs)

    # mask the encoder outputs for the 2nd attention block
    input_dec_padding_mask = tf.keras.layers.Lambda(
        create_padding_mask, output_shape=(1, 1, None),
        name='input_dec_padding_mask')(inputs)

    context_dec_padding_mask = tf.keras.layers.Lambda(
        create_padding_mask, output_shape=(1, 1, None),
        name='context_dec_padding_mask')(context)

    input_enc_outputs = encoder(
        vocab_size=vocab_size,
        num_layers=num_layers,
        units=units,
        d_model=d_model,
        num_heads=num_heads,
        dropout=dropout,
        name='inputs_enc'
    )(inputs=[inputs, input_enc_padding_mask])

    context_enc_outputs = encoder(
        vocab_size=vocab_size,
        num_layers=num_layers,
        units=units,
        d_model=d_model,
        num_heads=num_heads,
        dropout=dropout,
        name='context_enc'
    )(inputs=[context, context_enc_padding_mask])

    dec_outputs = decoder(
        vocab_size=vocab_size,
        num_layers=num_layers,
        units=units,
        d_model=d_model,
        num_heads=num_heads,
        dropout=dropout,
    )(inputs=[dec_inputs,
              input_enc_outputs,
              context_enc_outputs,
              look_ahead_mask,
              input_dec_padding_mask,
              context_dec_padding_mask
              ])

    outputs = tf.keras.layers.Dense(units=vocab_size, name='outputs_dense')(dec_outputs)
    outputs = tf.keras.layers.Softmax(axis=-1, name='outputs')(outputs)

    return tf.keras.Model(inputs=[inputs, context, dec_inputs], outputs=outputs, name=name)


def loss_function(max_length=32):
    def _loss_function(y_true, y_pred):
        y_true = tf.reshape(y_true, shape=(-1, max_length - 1))

        loss = tf.keras.losses.SparseCategoricalCrossentropy(
            from_logits=False, reduction='none')(y_true, y_pred)

        mask = tf.cast(tf.not_equal(y_true, 0), tf.float32)
        loss = tf.multiply(loss, mask)

        return tf.reduce_mean(loss)
    return _loss_function


class CustomSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
    def __init__(self, d_model, warmup_steps=4000):
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


def accuracy(max_length=32):
    def _accuracy(y_true, y_pred):
        # ensure labels have shape (batch_size, max_length - 1)
        y_true = tf.reshape(y_true, shape=(-1, max_length - 1))
        return tf.keras.metrics.sparse_categorical_accuracy(y_true, y_pred)
    return _accuracy


def make_model(
    tokenizer=None, 
    num_layers=2,
    units=512,
    d_model=256,
    num_heads=8,
    dropout=0.1,
    max_length=32,
    warmup_steps=4000):

    
    logging.info('Compiling model.')
    learning_rate = CustomSchedule(d_model, warmup_steps)

    optimizer = tf.keras.optimizers.Adam(
        learning_rate, beta_1=0.9, beta_2=0.98, epsilon=1e-9)

    # Vocabulary size plus start and end token
    vocab_size = tokenizer.vocab_size + 2

    if IS_TPU:
        with tpu_strategy.scope(): # creating the model in the TPUStrategy scope means we will train the model on the TPU
            model = transformer(
                vocab_size,
                num_layers,
                units,
                d_model,
                num_heads,
                dropout
                )

            model.compile(optimizer=optimizer, loss=loss_function(max_length),
                        metrics=[accuracy(max_length)])
    else:
        model = transformer(
                vocab_size,
                num_layers,
                units,
                d_model,
                num_heads,
                dropout
                )

        model.compile(optimizer=optimizer, loss=loss_function(max_length),
                        metrics=[accuracy(max_length)])

    return model


def save_obj(path, obj):
    with open(path, 'wb+') as f:
        pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)


def load_obj(path):
    with open(path, 'rb') as f:
        return pickle.load(f)


def load_model(model_opts):
    tokenizer = load_obj(tokenizer_path)
    # tokenizer = SubwordTextEncoder.load_from_file(tokenizer_path)

    logging.info('Loading model.')
    model = make_model(tokenizer, **model_opts)
    model.load_weights(model_weights_path)

    logging.info('Done!')
    return tokenizer, model

#@title Training, Evaluation & Inference Functions

def make_tokenizer(data, target_vocab_size=2**13):
    logging.info('Training tokenizer...')

    tokenizer = SubwordTextEncoder.build_from_corpus(data,
            target_vocab_size=target_vocab_size)

    logging.info(f'Target Tokenizer vocab size: {target_vocab_size}')
    logging.info(f'Tokenizer vocab size: {tokenizer.vocab_size}')

    # save tokenizer
    logging.info('Saving tokenizer.')

    if not os.path.exists(model_path):
        os.makedirs(model_path)

    save_obj(tokenizer_path, tokenizer)
    # tokenizer.save_to_file(tokenizer_path)

    logging.info('Done!')

    return tokenizer


def make_dataset(
    inputs,
    context,
    outputs,
    tokenizer=None,
    batch_size=128,
    buffer_size=20000,
    max_length=32,
    target_vocab_size=2**13):

    if not tokenizer:
        tokenizer = make_tokenizer(inputs + context + outputs, target_vocab_size)

    inputs, context, outputs = tokenize_and_filter(tokenizer,
                                                   inputs,
                                                   context,
                                                   outputs,
                                                   max_length
                                                   )

    logging.info('Making data set.')
    # decoder inputs use the previous target as input
    # remove START_TOKEN from targets
    dataset = tf.data.Dataset.from_tensor_slices((
        {
            'inputs': inputs,
            'context': context,
            'dec_inputs': outputs[:, :-1]
        },
        {
            'outputs': outputs[:, 1:]
        },
    ))

    dataset = dataset.cache() \
                    .shuffle(
                        buffer_size,
                        reshuffle_each_iteration=True) \
                    .batch(batch_size) \
                    .prefetch(tf.data.experimental.AUTOTUNE)

    return tokenizer, dataset


def train(model, train_data, eval_data, epochs=10, min_delta=0.001,
          patience=10, baseline=None):
    
    # reset session
    tf.keras.backend.clear_session()

    def _train(*callbacks):
        # training callbacks
        early_stopping = tf.keras.callbacks.EarlyStopping(
            monitor='loss', min_delta=min_delta,
            patience=patience, verbose=1,
            mode='auto', baseline=baseline,
            restore_best_weights=False
            )
        save_weights = tf.keras.callbacks.ModelCheckpoint(
            model_weights_path, monitor='loss',
            verbose=0, save_best_only=False,
            save_weights_only=True, mode='auto',
            save_freq='epoch'
        )
        terminate_on_nan = tf.keras.callbacks.TerminateOnNaN()

        # Create a callback that saves the model's weights
        logging.info('Training model.')
        try:
            model.fit(
                    train_data,
                    validation_data=eval_data,
                    validation_freq=5,
                    epochs=epochs,
                    callbacks=[
                        early_stopping,
                        terminate_on_nan,
                        # save_weights,
                        *callbacks
                        ]
                    )

        except KeyboardInterrupt:
            logging.info('\nTraining Interruped!')

        finally:
            logging.info('Saving model.')
            model.save_weights(model_weights_path, overwrite=True)

        return model
        
    history = []
    if IS_COLAB:
        with output.use_tags('train'):
            def lambdaCallback(epoch, logs):
                history.append(logs)
                if epoch % 5 == 0:
                    output.clear(output_tags='train')

            save_history = tf.keras.callbacks.LambdaCallback(
                on_epoch_end=lambdaCallback
                )
            model = _train(save_history)
    else:
        def lambdaCallback(epoch, logs):
            history.append(logs)

        save_history = tf.keras.callbacks.LambdaCallback(
            on_epoch_end=lambdaCallback
            )
        model = _train(save_history)

    return model, history


def evaluate(tokenizer, model, sentence, context, max_length, training=False):
    start_token, end_token = [tokenizer.vocab_size], [tokenizer.vocab_size + 1]

    sentence = preprocess_sentence(sentence)
    sentence = tf.expand_dims(start_token + tokenizer.encode(sentence) + end_token, axis=0)
    context = preprocess_sentence(context)
    context = tf.expand_dims(start_token + tokenizer.encode(context) + end_token, axis=0)
    output = tf.expand_dims(start_token, 0)

    for i in range(max_length):
        predictions = model(inputs=[sentence, context, output], training=training)

        # select the last word from the seq_len dimension
        predictions = predictions[:, -1:, :]
        predicted_id = tf.cast(tf.argmax(predictions, axis=-1), tf.int32)

        # return the result if the predicted_id is equal to the end token
        if tf.equal(predicted_id, end_token[0]):
            break

        # concatenated the predicted_id to the output which is given to the decoder
        # as imodel_optsts input.
        output = tf.concat([output, predicted_id], axis=-1)

    return tf.squeeze(output, axis=0)


def predict(tokenizer, model, sentence, context, max_length, training=False):
    prediction = evaluate(tokenizer, model, sentence, context, max_length, training)
    prediction = [i for i in prediction if i < tokenizer.vocab_size]
    return tokenizer.decode(prediction)

#@title Decoder Model Visualization

sample_decoder_layer = decoder_layer(
    units=128,
    d_model=64,
    num_heads=2,
    dropout=0.3,
    name="sample_decoder_layer")

tf.keras.utils.plot_model(
    sample_decoder_layer, to_file='decoder_layer.png', show_shapes=True)

#@title Transformer Model Visualization

# train_inputs, train_context, train_outputs, eval_inputs, eval_context, \
#  eval_outputs = load_conversations(1000)
# print(len(train_inputs))

# BUFFER_SIZE = 16
# BATCH_SIZE = 1
# VOCAB_SIZE = 2**12
# MAX_LENGTH = 64

# tokenizer = make_tokenizer(train_inputs + train_context + train_outputs, VOCAB_SIZE)

# inputs, context, outputs = tokenize_and_filter(tokenizer,
#                                       train_inputs,
#                                       train_context,
#                                       train_outputs,
#                                       MAX_LENGTH
#                                       )

# logging.info('Making data set.')
# # decoder inputs use the previous target as input
# # remove START_TOKEN from targets
# dataset = tf.data.Dataset.from_tensor_slices((
#     {
#         'inputs': inputs,
#         'context': context,
#         'dec_inputs': outputs[:, :-1]
#     },
#     {
#         'outputs': outputs[:, 1:]
#     },
# ))

# dataset = dataset.cache() \
#                 .shuffle(BUFFER_SIZE) \
#                 .batch(BATCH_SIZE) \
#                 .prefetch(tf.data.experimental.AUTOTUNE)


# len(list(dataset.as_numpy_iterator()))
# list(dataset.as_numpy_iterator())

sample_transformer = transformer(
    vocab_size=8192,
    num_layers=2,
    units=128,
    d_model=64,
    num_heads=4,
    dropout=0.3,
    name="sample_transformer"
)

tf.keras.utils.plot_model(
    sample_transformer, to_file='transformer.png', show_shapes=True)

#@title Training Routine { vertical-output: true, display-mode: "form" }

NEW_MODEL = True  #@param {type:"boolean"}
TRAIN_MODEL = True  #@param {type:"boolean"}
TRAIN_TOKENIZER = False  #@param {type:"boolean"}

corpus_name = "friends-corpus, movie-corpus, reddit-corpus-small" #@param {type:"string"}

# Training params
EPOCHS =  100#@param {type:"integer"}
if IS_TPU:
    BATCH_SIZE = 128 * tpu_strategy.num_replicas_in_sync
else:
    BATCH_SIZE = 128
BUFFER_SIZE = 100000  #@param {type:"integer"}
EVAL_PERCENT = 0.05  #@param {type:"number"}
WARMUP_STEPS = 2000  #@param {type:"integer"}
MIN_DELTA = 0.0005 #@param {type:"number"}
PATIENCE = 30  #@param {type:"integer"}
BASELINE = 0  #@param {type:"number"}
if not BASELINE:
    BASELINE = None

# tokenizer params
TARGET_VOCAB_SIZE = 2**13  #@param {type:"raw"}

# Maximum number of samples to preprocess
MAX_LENGTH =    16#@param {type:"integer"}
MAX_SAMPLES = 9999999  #@param {type:"integer"}

# Hyper-parameters
NUM_LAYERS =   4#@param {type:"integer"}
D_MODEL = 128  #@param {type:"integer"}
NUM_HEADS = 8  #@param {type:"integer"}
UNITS = 128  #@param {type:"integer"}
DROPOUT = 0.1  #@param {type:"number"}

train_questions = []
train_context = []
train_answers = []
eval_questions = []
eval_context = []
eval_answers = []
for corpus in corpus_name.split(', '):
    corpus_tuple = load_conversations(corpus, MAX_SAMPLES, EVAL_PERCENT)
    train_questions.extend(corpus_tuple[0])
    train_context.extend(corpus_tuple[1])
    train_answers.extend(corpus_tuple[2])
    eval_questions.extend(corpus_tuple[3])
    eval_context.extend(corpus_tuple[4])
    eval_answers.extend(corpus_tuple[5])

print(f'Train questions len: {len(train_questions)}')
print(f'Train context len: {len(train_context)}')
print(f'Train answers len: {len(train_answers)}')

print(f'Eval questions len: {len(eval_questions)}')
print(f'Eval context len: {len(eval_context)}')
print(f'Eval answers len: {len(eval_answers)}')
 
if NEW_MODEL:

    train_opts = {
        'epochs': EPOCHS,
        'patience': PATIENCE,
        'min_delta': MIN_DELTA,
        'baseline': BASELINE
    }
    
    dataset_opts = {
        'batch_size': BATCH_SIZE,
        'buffer_size': BUFFER_SIZE,
        'max_length': MAX_LENGTH,
        'target_vocab_size': TARGET_VOCAB_SIZE
    }
    
    model_opts = {
        'num_layers': NUM_LAYERS,
        'units': UNITS,
        'd_model': D_MODEL,
        'num_heads': NUM_HEADS,
        'dropout': DROPOUT,
        'max_length': MAX_LENGTH,
        'warmup_steps': WARMUP_STEPS,
    }

    save_obj(model_config_path, model_opts)
    save_obj(dataset_config_path, dataset_opts)
    save_obj(train_config_path, train_opts)

    tokenizer = None
    if not TRAIN_TOKENIZER:
        tokenizer = load_obj(tokenizer_path)

    tokenizer, train_data = make_dataset(
        train_questions,
        train_context,
        train_answers,
        tokenizer,
        **dataset_opts
        )
    _, eval_data = make_dataset(
        eval_questions,
        eval_context,
        eval_answers,
        tokenizer,
        **dataset_opts
        )
    
    model = make_model(tokenizer, **model_opts)

    if TRAIN_MODEL:
        model, history = train(model, train_data,
                               eval_data, **train_opts)
else:
    train_opts = load_obj(train_config_path) 
    dataset_opts = load_obj(dataset_config_path)
    model_opts = load_obj(model_config_path)

    tokenizer, model = load_model(model_opts)

    if TRAIN_MODEL:
        tokenizer, train_data = make_dataset(
            train_questions,
            train_context,
            train_answers,
            tokenizer,
            **dataset_opts
            )
        _, eval_data = make_dataset(
            eval_questions,
            eval_context,
            eval_answers,
            tokenizer,
            **dataset_opts)
        
        model, history = train(model, train_data,
                               eval_data, **train_opts)


model.summary()

context = 'Welcome to the desert of the real.'
for you in ['am I dead?', 'What is the matrix?', "I thought it wasn't real"]:
    prediction = context = predict(tokenizer, model, you, context, max_length=MAX_LENGTH)
    print(f'transformer: {prediction}')

hist_df = pd.DataFrame.from_records(history)
hist_df['epoch'] = [i for i in range(len(history))]

graphs = ['loss', 'val_loss', '_accuracy', 'val__accuracy']
def make_graph(y):
    return alt.Chart(hist_df).mark_point().encode(
        x='epoch',
        y=y,
    ).properties(
        width=200,
        height=200
    )

alt.hconcat(*[make_graph(y) for y in graphs])

#@title Training History Compare { vertical-output: true }
model.summary()

hist_df = pd.DataFrame.from_records(history)
hist_df['epoch'] = [i for i in range(len(history))]

graphs = ['loss', 'val_loss', '_accuracy', 'val__accuracy']
def make_graph(y):
    return alt.Chart(hist_df).mark_point().encode(
        x='epoch',
        y=y,
    ).properties(
        width=160,
        height=160
    )

alt.hconcat(*[make_graph(y) for y in graphs])

#@title Talk to the model { vertical-output: true }
context = "The pill you took is part of a trace program. It's design to disrupt your input/output carrier signal so we can pinpoint your location." #@param {type:"string"}
you = " What does that mean?" #@param {type:"string"}

prediction = context = predict(tokenizer, model, you, context, max_length=MAX_LENGTH)
print(f'transformer: {prediction}')

#@title Self Context
you = "are we dead?" #@param {type:"string"}
prediction = context = predict(tokenizer, model, you, context, max_length=MAX_LENGTH)
print(f'transformer: {prediction}')