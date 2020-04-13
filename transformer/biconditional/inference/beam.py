import logging
import tensorflow as tf
from ..preprocessing import preprocess_sentence
from ...utils.inference import Beam


logging.basicConfig(level=logging.INFO)


def evaluate_beam(tokenizer, model, sentence, context, max_length, beam_size, max_hypothesis, max_depth, training):
    start_token, end_token = [tokenizer.vocab_size], [tokenizer.vocab_size + 1]

    sentence = preprocess_sentence(sentence)
    sentence = tf.expand_dims(start_token + tokenizer.encode(sentence) + end_token, axis=0)
    context = preprocess_sentence(context)
    context = tf.expand_dims(start_token + tokenizer.encode(context) + end_token, axis=0)

    beam = Beam(model, start_token, end_token, max_length, beam_size)

    predictions = beam.run([sentence, context], max_hypothesis, max_depth)

    return predictions


def predict_beam(tokenizer, model, sentence, context, max_length=None, beam_size=8, max_hypothesis=4,
        max_depth=32, training=False, show_hypotheses=True):

    predictions = evaluate_beam(tokenizer, model, sentence, context, max_length, beam_size, max_hypothesis,
            max_depth, training)

    if show_hypotheses:
        for prediction in predictions:
            sentence = [i for i in prediction['path'] if i < tokenizer.vocab_size]
            result = tokenizer.decode(sentence)
            logging.info(f"r: {result} [{prediction['score']}]")

    return tokenizer.decode([i for i in predictions[0]['path'] if i < tokenizer.vocab_size])
