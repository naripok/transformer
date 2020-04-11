import logging
import tensorflow as tf
from ..preprocessing import preprocess_sentence


class Beam(object):
    def __init__(self, model, sentence, context, start_token, end_token, max_length=None, beam_size=2):
        self.beam_size = beam_size
        self.leaves = [tf.expand_dims(start_token, 0)]
        self.l_scores = [tf.expand_dims([tf.constant(0.)], 0)]
        self.paths  = []
        self.p_scores = []
        self.model = model
        self.sentence = sentence
        self.context = context
        self.end_token = end_token
        self._step = 0
        self.max_length = max_length

    def add_node(self, path, score, node, node_score):
        return tf.concat([path, node], axis=-1), tf.concat([score, tf.add(score[:, -1], node_score)], axis=-1)

    def add_path(self, path, score):
        self.paths.append(path)
        self.p_scores.append(score)

    def set_leaf(self, paths, scores):
        self.leaves = paths
        self.l_scores = scores

    def eval_path(self, path, training=False):
        # shape (batch_size, sentence_size, vocab_size)
        predictions = self.model(inputs=[self.sentence, self.context, path], training=training)

        # select the last word from the seq_len dimension
        # shape (batch_size, 1, vocab_size)
        predictions = predictions[:, -1:, :]

        # select elements with top K probabilities
        top_args = tf.math.top_k(predictions, k=self.beam_size)

        predicted_ids = tf.cast(top_args.indices, tf.int32)  # shape (batch, 1, size)
        predicted_scores = tf.cast(tf.map_fn(tf.math.log, top_args.values), tf.float32)  # shape (batch, 1, size)

        return predicted_ids, predicted_scores

    def choose_leaf(self, leaves, scores):
        rank = sorted(
                [{'sentence': l, 'score': s} for l, s in zip(leaves, scores)],
                key=lambda x: x['score'][:, -1],
                reverse=True
                )
        return [l['sentence'] for l in rank][:self.beam_size], [s['score'] for s in rank][:self.beam_size]

    def step(self, max_hypothesis):
        results, r_scores = [], []

        choosen_leaves, choosen_scores = self.choose_leaf(self.leaves, self.l_scores)

        while choosen_leaves:
            path, score = choosen_leaves.pop(), choosen_scores.pop()

            node_ids, node_r_scores = self.eval_path(path)

            semi_results, semi_r_scores = [], []
            for i in range(self.beam_size):

                exp_path, exp_score = self.add_node(path, score, node_ids[:, :, i], node_r_scores[:, :, i])

                if tf.equal(node_ids[:, :, i], self.end_token[0]) \
                or (self.max_length and exp_path.shape[-1] >= self.max_length):
                    self.add_path(exp_path, exp_score)
                    continue

                semi_results.append(exp_path)
                semi_r_scores.append(exp_score)

            results.extend(semi_results)
            r_scores.extend(semi_r_scores)

        self.set_leaf(results, r_scores)
        self._step += 1

        return results, r_scores

    def expand(self, max_hypothesis, max_depth):
        while self.leaves and self._step <= max_depth:
            self.step(max_hypothesis)

    def get_hypothesis_scores(self):
        return [tf.squeeze(tf.math.divide(
            s[:, -1],
            s.shape[-1]), axis=0)
            for s in self.p_scores]

    def get_hypothesis(self):
        return [tf.squeeze(p, axis=0) for p in self.paths]

    def run(self, max_hypothesis=8, max_depth=32):
        self.expand(max_hypothesis, max_depth)
        return sorted([
            {'sentence': l, 'score': s}
            for l, s in zip(self.get_hypothesis(), self.get_hypothesis_scores())],
            key=lambda x: x['score'],
            reverse=False
            )[-max_hypothesis:]


def evaluate_beam(tokenizer, model, sentence, context, max_length, beam_size, max_hypothesis, max_depth, training):
    start_token, end_token = [tokenizer.vocab_size], [tokenizer.vocab_size + 1]

    sentence = preprocess_sentence(sentence)
    sentence = tf.expand_dims(start_token + tokenizer.encode(sentence) + end_token, axis=0)
    context = preprocess_sentence(context)
    context = tf.expand_dims(start_token + tokenizer.encode(context) + end_token, axis=0)

    beam = Beam(model, sentence, context, start_token, end_token, max_length, beam_size)

    predictions = beam.run(max_hypothesis, max_depth)

    return predictions


def predict_beam(tokenizer, model, sentence, context, max_length=None, beam_size=16, max_hypothesis=8,
        max_depth=32, training=False):
    predictions = evaluate_beam(tokenizer, model, sentence, context, max_length, beam_size, max_hypothesis,
            max_depth, training)
    for prediction in predictions:
        sentence = [i for i in prediction['sentence'] if i < tokenizer.vocab_size]
        result = tokenizer.decode(sentence)
        logging.info(f"r: {result} [{prediction['score']}]")

    return result
