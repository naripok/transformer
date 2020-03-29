from .core import ChatBot
from .preprocessing import load_conversations
import json
import os

if __name__ == "__main__":
    import logging
    logging.basicConfig(level=logging.DEBUG)

    LOAD_MODEL = True
    TRAIN_TOKENIZER = False
    TRAIN_MODEL = True
    DOWNLOAD_DATA = False
    NUM_EPOCHS = 200
    NUM_POSTS = 100
    NUM_COMMENTS = 20
    PATIENCE = 20
    MIN_DELTA = 0.001
    BASELINE = 0.01
    MAX_SAMPLES = 50000

    data_dir = './data'
    data_path = data_dir + '/data.json'

    logging.info('Training routine initiated...')

    if DOWNLOAD_DATA:
        try:
            inputs, outputs = load_conversations(MAX_SAMPLES)
        except KeyboardInterrupt:
            logging.info('Interrupted!')
        finally:
            data = []
            for x, y in zip(inputs, outputs):
                data += [x, y]

            logging.info('Saving data to disk.')
            if not os.path.exists(data_dir):
                os.makedirs(data_dir)

            with open(data_path, 'w+') as f:
                json.dump(data, f)

    else:
        logging.info('Loading data from disk.')
        try:
            with open(data_path, 'r') as f:
                data = json.load(f)
        except FileNotFoundError as e:
            logging.info('No data found on disk. You need to download the data first.')
            raise

    logging.info(f'N samples: {len(data)}')

    logging.info('Instantiating model...')

    bot = ChatBot('bot', load=LOAD_MODEL)

    if TRAIN_TOKENIZER and not LOAD_MODEL:
        logging.info('Training tokenizer...')
        bot.train_tokenizer(data)

    if TRAIN_MODEL:
        logging.info('Training model...')
        bot.train(data, epochs=NUM_EPOCHS, min_delta=MIN_DELTA, patience=PATIENCE)

    logging.info('Model responses examples:\n')

    # feed the model with its previous output
    sentence = 'I am not crazy, my mother had me tested.'
    print(sentence)

    while True:
        try:
            sentence = bot.get_response(input())
            print(sentence)
        except KeyboardInterrupt:
            break
