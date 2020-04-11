import logging
import tensorflow as tf
from ..preprocessing import preprocess_sentence


def evaluate_greedy(tokenizer, model, sentence, context, max_length, training=False):
    start_token, end_token = [tokenizer.vocab_size], [tokenizer.vocab_size + 1]

    sentence = preprocess_sentence(sentence)
    sentence = tf.expand_dims(start_token + tokenizer.encode(sentence) + end_token, axis=0)
    context = preprocess_sentence(context)
    context = tf.expand_dims(start_token + tokenizer.encode(context) + end_token, axis=0)
    output = tf.expand_dims(start_token, 0)

    while True:
        predictions = model(inputs=[sentence, context, output], training=training)

        # select the last word from the seq_len dimension
        predictions = predictions[:, -1:, :]
        predicted_id = tf.cast(tf.argmax(predictions, axis=-1), tf.int32)

        # return the result if the predicted_id is equal to the end token
        if tf.equal(predicted_id, end_token[0]) \
        or (max_length and output.shape[-1] >= max_length):
            break

        # concatenated the predicted_id to the output which is given to the decoder
        # as imodel_optsts input.
        output = tf.concat([output, predicted_id], axis=-1)

    return tf.squeeze(output, axis=0)


def predict_greedy(tokenizer, model, sentence, context, max_length=None, training=False):
    prediction = evaluate_greedy(tokenizer, model, sentence, context, max_length, training)
    prediction = [i for i in prediction if i < tokenizer.vocab_size]
    return tokenizer.decode(prediction)

