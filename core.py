from tqdm import tqdm
import logging
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


if __name__ == '__main__':
    from .model import load_conversations

    LOAD_MODEL = True
    TRAIN_TOKENIZER = False
    TRAIN_MODEL = True
    MAX_SAMPLES = 50000

    bot = ChatBot('bot', load=LOAD_MODEL)

    inputs, outputs = load_conversations(MAX_SAMPLES)
    data = []
    for x, y in zip(inputs, outputs):
        data += [x, y]

    logging.info(f'N samples: {len(data)}')

    if TRAIN_TOKENIZER and not LOAD_MODEL:
        bot.train_tokenizer(data)

    if TRAIN_MODEL:
        bot.train(data, epochs=10)

    # feed the model with its previous output
    sentence = 'I am not crazy, my mother had me tested.'
    print(sentence)

    for _ in range(10):
        sentence = bot.get_response(sentence)
        print(sentence)
