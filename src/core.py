import logging
from .model import make_dataset, make_model, \
        load_model, predict, train, make_tokenizer
import random as rd


class ChatBot(object):
    def __init__(self, name):
        self.name = name
        self.tokenizer = None
        self.model = None
        self.buffer = ''

        if load:
            try:
                model_opts = load_obj(model_config_path)
                self.tokenizer, self.model = load_model(model_opts)
            except OSError:
                raise BaseException('Model not found.')

    def get_response(self, sentence):
        if not self.model:
            return 'You need to train the model first.'
        return predict(self.tokenizer, self.model, sentence, self.bufer,
                max_length=MAX_LENGTH)

    def sample_tokenizer(self, num_samples, random=False):
        if not self.tokenizer:
            return 'You need to train the tokenizer first.'

        if random:
            return self.tokenizer.decode(
                    [rd.randint(self.tokenizer.num_words)
                        for _ in range(num_samples)
                        ]
                    )
        else:
            return self.tokenizer.decode(
                    [i for i in range(num_samples)]
                    )
