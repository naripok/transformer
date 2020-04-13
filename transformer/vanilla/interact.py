import logging
import tensorflow as tf
from .preprocessing import load_conversations, tokenize_and_filter
from .inference import predict_greedy, predict_beam
from .train import load_model
from .model import *
from .params import *
from ..components import load_obj


logging.basicConfig(level=logging.INFO)


train_opts = load_obj(train_config_path)
dataset_opts = load_obj(dataset_config_path)
model_opts = load_obj(model_config_path)

tokenizer, model = load_model(model_opts)

if __name__ == "__main__":
    context = 'Hi!'
    print(f'Transformer: {context}')

    while True:
        try:
            you = input('\nYou: ')
            prediction = predict_beam(tokenizer, model, you)
            print(f'Transformer: {prediction}')
        except KeyboardInterrupt:
            print('Bye...')
            break
