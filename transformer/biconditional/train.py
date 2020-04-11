import os
import datetime
import logging
import re
import pickle
import tensorflow as tf
from tensorflow_datasets.core.features.text import SubwordTextEncoder
import pandas as pd
import altair as alt
from .preprocessing import load_conversations, tokenize_and_filter
from .inference import predict_greedy, predict_beam
from .model import *


logging.basicConfig(level=logging.INFO)
tf.random.set_seed(42)
alt.renderers.enable('altair_viewer')
tf.keras.backend.clear_session()


IS_COLAB = False #@param {type:"boolean"}
MOUNT_DRIVE = False #@param {type:"boolean"}
IS_TPU = False

NEW_MODEL = True  #@param {type:"boolean"}
TRAIN_MODEL = True  #@param {type:"boolean"}
TRAIN_TOKENIZER = True  #@param {type:"boolean"}

corpus_name = "friends-corpus, movie-corpus, reddit-corpus-small" #@param {type:"string"}

# Training params
EPOCHS = 100
if IS_TPU:
    BATCH_SIZE = 128 * tpu_strategy.num_replicas_in_sync
else:
    BATCH_SIZE = 128
BUFFER_SIZE = 100000
EVAL_PERCENT = 0.05
WARMUP_STEPS = 2000
MIN_DELTA = 0.0005
PATIENCE = 30
BASELINE = 0
if not BASELINE:
    BASELINE = None

# tokenizer params
TARGET_VOCAB_SIZE = 2**13

# Maximum number of samples to preprocess
MAX_LENGTH = 16
MAX_SAMPLES = 9999999

# Hyper-parameters
NUM_LAYERS = 2
D_MODEL = 128
NUM_HEADS = 8
UNITS = 128
DROPOUT = 0.1


if IS_COLAB:
    from google.colab import output
    try:
        with output.use_tags('setup'):
            #  !pip install convokit
            #  !python3 -m spacy download en
            tpu = tf.distribute.cluster_resolver.TPUClusterResolver()  # TPU detection
            print('Running ovariable_namen TPU ', tpu.cluster_spec().as_dict()['worker'])
            tf.config.experimental_connect_to_cluster(tpu)
            tf.tpu.experimental.initialize_tpu_system(tpu)
            tpu_strategy = tf.distribute.experimental.TPUStrategy(tpu)
            IS_TPU = True
        output.clear(output_tags='setup')

    except ValueError:
        logging.info('Not connected to a TPU runtime')

if IS_COLAB and MOUNT_DRIVE:
    from google.colab import drive
    drive.mount('/content/drive', force_remount=True)

if IS_COLAB:
    model_path = "/content/drive/My Drive/discordbot/saved_model"  #@param {type:"string"}
else:
    model_path = "./saved_model/biconditional"  #@param {type:"string"}

if not os.path.exists(model_path):
    os.makedirs(model_path)

model_weights_path = model_path + '/weights.h5'
tokenizer_path = model_path + '/saved_tokenizer.pickle'
model_config_path = model_path + '/model_config.pickle'
dataset_config_path = model_path + '/dataset_config.pickle'
train_config_path = model_path + '/train_config.pickle'
log_dir = model_path + '/logs/fit/' + datetime.datetime.now().strftime('%Y%m%d-%H%M%S')


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


if __name__ == "__main__":

    if IS_TPU:
        BATCH_SIZE = BATCH_SIZE * tpu_strategy.num_replicas_in_sync
    else:
        BATCH_SIZE = BATCH_SIZE

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

        logging.info(f'Train questions len: {len(train_questions)}')
        logging.info(f'Train context len: {len(train_context)}')
        logging.info(f'Train answers len: {len(train_answers)}')

        logging.info(f'Eval questions len: {len(eval_questions)}')
        logging.info(f'Eval context len: {len(eval_context)}')
        logging.info(f'Eval answers len: {len(eval_answers)}')

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

            alt.hconcat(*[make_graph(y) for y in graphs]).show()

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

            alt.hconcat(*[make_graph(y) for y in graphs]).show()


    context = 'Welcome to the desert of the real.'
    for you in ['am I dead?', 'What is the matrix?', "I thought it wasn't real"]:
        prediction = context = predict_greedy(tokenizer, model, you, context, max_length=MAX_LENGTH)
        print(f'transformer: {prediction}')

    #@title Talk to the model { vertical-output: true }
    context = "The pill you took is part of a trace program. It's design to disrupt your input/output carrier signal so we can pinpoint your location." #@param {type:"string"}
    you = " What does that mean?" #@param {type:"string"}

    prediction = context = predict_beam(tokenizer, model, you, context, beam_size=1)
    print(f'transformer: {prediction}')

    #@title Self Context
    you = "are we dead?" #@param {type:"string"}
    prediction = context = predict_beam(tokenizer, model, you, context)
    print(f'transformer: {prediction}')
