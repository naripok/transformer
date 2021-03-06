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

from __future__ import absolute_import, division, print_function, unicode_literals

import logging
import tensorflow as tf
from ..components import *


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
