import os
import argparse
import datetime
import logging
import tensorflow as tf
from .preprocessing import load_conversations, tokenize_and_filter
from .inference import predict_greedy, predict_beam
from .model import transformer, CustomSchedule, loss_function, accuracy
from .params import *
from ..components import save_obj, load_obj, make_tokenizer, train


parser = argparse.ArgumentParser(
        description='Train the vanilla transformer model on diferent datasets')
parser.add_argument('--new',
        action='store_true', help='Train model form scratch')
parser.add_argument('--train-model',
        action='store_true', help='Train model')
parser.add_argument('--train-tokenizer',
        action='store_true', help='Train tokenizer')
parser.add_argument('--colab',
        action='store_true', help='Colab environment')
parser.add_argument('--epochs', default=3, type=int,
        help='Training epochs')
parser.add_argument('--batch-size', default=128, type=int,
        help='Batch size for training')
parser.add_argument('--max-samples', default=1000000, type=int,
        help='Max data point to use on dataset')
parser.add_argument('--buffer-size', default=100000, type=int,
        help='Buffer size for shuffling')
parser.add_argument('--eval-percent', default=0.1, type=float,
        help='Percentage of dataset for evaluation split')
parser.add_argument('--warmup-steps', default=4000, type=int,
        help='Warm steps for LR scheduling')
parser.add_argument('--min-delta', default=0.005, type=float,
        help='Min delta for early stopping')
parser.add_argument('--patience', default=5, type=int,
        help='Patience for early stopping')
parser.add_argument('--baseline', default=0, type=float,
        help='Baseline for early stopping')
parser.add_argument('--corpus',
        default='friends-corpus, movie-corpus, reddit-corpus-small',
        type=str, help='Comma separated corpus names')
parser.add_argument('--vocab-size', default=2**13, type=float,
        help='Target vocabulary size')
parser.add_argument('--max-length', default=32, type=int,
        help='Max sentence length to process')
parser.add_argument('--dropout', default=0.1, type=float,
        help='Training dropout normalization')
parser.add_argument('--units', default=256, type=int,
        help='Hidden units')
parser.add_argument('--num-layers', default=2, type=int,
        help='Num layers')
parser.add_argument('--num-heads', default=8, type=int,
        help='Num attention heads ')
parser.add_argument('--d-model', default=256, type=int,
        help='Dense model')
args = parser.parse_args()


logging.basicConfig(level=logging.INFO)
tf.random.set_seed(42)
tf.keras.backend.clear_session()

IS_COLAB = args.colab

NEW_MODEL = args.new
TRAIN_MODEL = args.train_model
TRAIN_TOKENIZER = args.train_tokenizer

CORPUS_NAME = args.corpus

# Training params
EPOCHS = args.epochs
MAX_SAMPLES = args.max_samples
BATCH_SIZE = args.batch_size
BUFFER_SIZE = args.buffer_size
EVAL_PERCENT = args.eval_percent
WARMUP_STEPS = args.warmup_steps
MIN_DELTA = args.min_delta
PATIENCE = args.patience
BASELINE = args.baseline
if not BASELINE:
    BASELINE = None

# tokenizer params
TARGET_VOCAB_SIZE = args.vocab_size

# Maximum number of samples to preprocess
MAX_LENGTH = args.max_length

# Hyper-parameters
NUM_LAYERS = args.num_layers
D_MODEL = args.d_model
NUM_HEADS = args.num_heads
UNITS = args.units
DROPOUT = args.dropout


IS_TPU = False
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


def load_model(model_opts):
    tokenizer = load_obj(tokenizer_path)
    # tokenizer = SubwordTextEncoder.load_from_file(tokenizer_path)

    logging.info('Loading model.')
    model = make_model(tokenizer, **model_opts)
    model.load_weights(model_weights_path)

    logging.info('Done!')
    return tokenizer, model


def make_dataset(
    inputs,
    outputs,
    tokenizer=None,
    batch_size=128,
    buffer_size=20000,
    max_length=32,
    target_vocab_size=2**13):

    if not tokenizer:
        tokenizer = make_tokenizer(inputs + outputs, target_vocab_size)
        logging.info('Saving tokenizer.')
        save_obj(tokenizer_path, tokenizer)
        # tokenizer.save_to_file(tokenizer_path)

    inputs, outputs = tokenize_and_filter(
            tokenizer,
            inputs,
            outputs,
            max_length
            )

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
                    .shuffle(
                            buffer_size,
                            reshuffle_each_iteration=True) \
                    .batch(batch_size) \
                    .prefetch(tf.data.experimental.AUTOTUNE)

    return tokenizer, dataset


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
        train_answers = []
        eval_questions = []
        eval_answers = []

        for corpus in CORPUS_NAME.split(', '):
            corpus_tuple = load_conversations(corpus, MAX_SAMPLES, EVAL_PERCENT)

            train_questions.extend(corpus_tuple[0])
            train_answers.extend(corpus_tuple[1])
            eval_questions.extend(corpus_tuple[2])
            eval_answers.extend(corpus_tuple[3])

        logging.info(f'Train questions len: {len(train_questions)}')
        logging.info(f'Train answers len: {len(train_answers)}')
        logging.info(f'Eval questions len: {len(eval_questions)}')
        logging.info(f'Eval answers len: {len(eval_answers)}')

        tokenizer, train_data = make_dataset(
            train_questions,
            train_answers,
            **dataset_opts
            )
        _, eval_data = make_dataset(
            eval_questions,
            eval_answers,
            tokenizer,
            **dataset_opts
            )

        model = make_model(tokenizer, **model_opts)

        if TRAIN_MODEL:
            model = train(model, train_data,
                    eval_data, **train_opts,
                    save_path=model_weights_path)
    else:
        train_opts = load_obj(train_config_path)
        dataset_opts = load_obj(dataset_config_path)
        model_opts = load_obj(model_config_path)
        tokenizer, model = load_model(model_opts)

        if TRAIN_MODEL:

            train_questions = []
            train_answers = []
            eval_questions = []
            eval_answers = []
            for corpus in CORPUS_NAME.split(', '):
                corpus_tuple = load_conversations(corpus, MAX_SAMPLES, EVAL_PERCENT)
                train_questions.extend(corpus_tuple[0])
                train_answers.extend(corpus_tuple[1])
                eval_questions.extend(corpus_tuple[2])
                eval_answers.extend(corpus_tuple[3])

            logging.info(f'Train questions len: {len(train_questions)}')
            logging.info(f'Train answers len: {len(train_answers)}')
            logging.info(f'Eval questions len: {len(eval_questions)}')
            logging.info(f'Eval answers len: {len(eval_answers)}')

            tokenizer, train_data = make_dataset(
                train_questions,
                train_answers,
                **dataset_opts
                )
            _, eval_data = make_dataset(
                eval_questions,
                eval_answers,
                tokenizer,
                **dataset_opts)

            model = train(model, train_data,
                    eval_data, **train_opts,
                    save_path=model_weights_path)

            model.summary()

    for you in ['am I dead?', 'What is the matrix?', "I thought it wasn't real"]:
        prediction = predict_greedy(tokenizer, model, you, max_length=MAX_LENGTH)
        print(f'transformer: {prediction}')

    you = "what is your name?"
    print(f'you: {you}')
    print(f'transformer: {predict_beam(tokenizer, model, you, max_length=MAX_LENGTH)}')
