import os
import re
import logging
from tqdm import tqdm
import tensorflow as tf
import convokit
from convokit import Corpus, download
import nltk; nltk.download('punkt')
from ..utils.preprocessing import preprocess_sentence


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


