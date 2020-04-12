import os
import datetime
import tensorflow as tf
#  import pandas as pd
from tensorflow_datasets.core.features.text import SubwordTextEncoder
#  import altair as alt
import pickle
from .model import *
from .preprocessing import *
from .inference import predict_beam, predict_greedy

#  alt.renderers.enable('altair_viewer')


NEW_MODEL = True  #@param {type:"boolean"}
TRAIN = True  #@param {type:"boolean"}
IS_TPU = False  #@param {type:"boolean"}

# Training params
EPOCHS = 2 #@param {type:"integer"}
MAX_SAMPLES = 100000  #@param {type:"integer"}
if IS_TPU:
    BATCH_SIZE = 128 * tpu_strategy.num_replicas_in_sync
else:
    BATCH_SIZE = 32
BUFFER_SIZE = 5000 #@param {type:"integer"}
WARMUP_STEPS = 1000  #@param {type:"integer"}
MIN_DELTA = 0.001 #@param {type:"number"}
PATIENCE = 10  #@param {type:"integer"}
BASELINE = 0  #@param {type:"number"}
EVAL_PERCENT = 0.1  #@param {type:"number"}
if not BASELINE:
    BASELINE = None

# tokenizer params
TARGET_VOCAB_SIZE = 2**13  #@param {type:"raw"}

# Maximum number of samples to preprocess

# Hyper-parameters
MAX_LENGTH = 32  #@param {type:"integer"}
NUM_LAYERS = 2  #@param {type:"integer"}
D_MODEL = 64  #@param {type:"integer"}
NUM_HEADS = 8  #@param {type:"integer"}
UNITS = 128 #@param {type:"integer"}
DROPOUT = 0.1  #@param {type:"number"}

model_path = "./saved_model/vanilla"  #@param {type:"string"}
model_weights_path = model_path + '/weights.h5'
tokenizer_path = model_path + '/saved_tokenizer.pickle'
log_dir = model_path + '/logs/fit/' + datetime.datetime.now().strftime('%Y%m%d-%H%M%S')

if not os.path.exists(log_dir):
    os.makedirs(log_dir)


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
    logging.info('Loading tokenizer.')
    with open(tokenizer_path, 'rb') as f:
        tokenizer = pickle.load(f)

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

    with open(tokenizer_path, 'wb+') as f:
        pickle.dump(tokenizer, f, protocol=pickle.HIGHEST_PROTOCOL)

    # tokenizer.save_to_file(tokenizer_path)

    logging.info('Done!')

    return tokenizer


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

    inputs, outputs = tokenize_and_filter(tokenizer, inputs, outputs, max_length)

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
                    .shuffle(buffer_size) \
                    .batch(batch_size) \
                    .prefetch(tf.data.experimental.AUTOTUNE)

    return tokenizer, dataset


def train(model, train_data, eval_data, epochs=10, min_delta=0.001,
          patience=10, baseline=None):

    # reset session
    #  tf.keras.backend.clear_session()

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
                    epochs=epochs,
                    callbacks=[
                        early_stopping,
                        terminate_on_nan,
                        save_weights,
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

    corpus_name = "friends-corpus, movie-corpus, reddit-corpus-small" #@param {type:"string"}

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


    if NEW_MODEL:

        train_questions = []
        train_answers = []
        eval_questions = []
        eval_answers = []

        for corpus in corpus_name.split(', '):
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

        if TRAIN:
            model, history = train(model, train_data,
                                   eval_data, **train_opts)
    else:
        tokenizer, model = load_model(model_opts)

        if TRAIN:

            train_questions = []
            train_answers = []
            eval_questions = []
            eval_answers = []

            for corpus in corpus_name.split(', '):
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

            model, history = train(model, train_data,
                                   eval_data, **train_opts)

            model.summary()

            #  hist_df = pd.DataFrame.from_records(history)
            #  hist_df['epoch'] = [i for i in range(len(history))]
            #  
            #  graphs = ['loss', 'val_loss', '_accuracy', 'val__accuracy']
            #  def make_graph(y):
            #      return alt.Chart(hist_df).mark_point().encode(
            #          x='epoch',
            #          y=y,
            #      ).properties(
            #          width=360,
            #          height=360
            #      )
            #  
            #  alt.hconcat(*[make_graph(y) for y in graphs]).show()

    you = "what is your name?"
    print(f'you: {you}')
    print(f'transformer: {predict_beam(tokenizer, model, you)}')
