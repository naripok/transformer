import logging
from tqdm import tqdm
from .model import make_dataset, make_model, \
        load_model, predict, train, make_tokenizer
import random as rd


class ChatBot(object):
    def __init__(self, name, load=True):
        self.name = name
        self.tokenizer = None
        self.model = None

        if load:
            try:
                self.tokenizer, self.model = load_model()
            except OSError:
                logging.warn('Model path not found. Instantiating new model.')

    def get_response(self, sentence):
        if not self.model:
            return 'You need to train the model first.'
        return predict(self.tokenizer, self.model, sentence)

    def train(self, data, epochs, min_delta=0.01, patience=5, baseline=None):
        self.tokenizer, data = make_dataset(
                data[::2],
                data[1::2],
                tokenizer=self.tokenizer
                )
        self.tokenizer, self.model = train(self.tokenizer, self.model, data, epochs, min_delta,
                patience, baseline)

    def train_tokenizer(self, data):
        self.tokenizer = make_tokenizer(data)

    def sample_tokenizer_vocab(self, num_samples, random=False):
        if not self.tokenizer:
            return 'You need to train the tokenizer first.'

        if random:
            return self.tokenizer.sequence_to_text(
                    [rd.randint(self.tokenizer.num_words)
                        for _ in range(num_samples)
                        ]
                    )
        else:
            return self.tokenizer.sequence_to_text(
                    [i for i in range(num_samples)]
                    )
