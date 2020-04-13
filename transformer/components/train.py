import os
import logging
import tensorflow as tf
from tensorflow_datasets.core.features.text import SubwordTextEncoder
from .serialization import save_obj, load_obj


def make_tokenizer(data, target_vocab_size=2**13):
    logging.info('Training tokenizer...')

    tokenizer = SubwordTextEncoder.build_from_corpus(data,
            target_vocab_size=target_vocab_size)

    logging.info(f'Target Tokenizer vocab size: {target_vocab_size}')
    logging.info(f'Tokenizer vocab size: {tokenizer.vocab_size}')

    logging.info('Done!')

    return tokenizer


def train(model, train_data, eval_data, epochs=10, min_delta=0.001,
          patience=10, baseline=None, save_path='./saved_model'):

    # reset session
    tf.keras.backend.clear_session()

    # training callbacks
    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor='loss', min_delta=min_delta,
        patience=patience, verbose=1,
        mode='auto', baseline=baseline,
        restore_best_weights=False
        )
    save_weights = tf.keras.callbacks.ModelCheckpoint(
        save_path, monitor='loss',
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
                    save_weights,
                    ]
                )

    except KeyboardInterrupt:
        logging.info('\nTraining Interruped!')

    finally:
        logging.info('Saving model.')
        model.save_weights(save_path, overwrite=True)

    return model
