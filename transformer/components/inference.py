import tensorflow as tf

class Beam(object):
    def __init__(self, model, start_token, end_token, max_length=None, beam_size=2):
        self.model = model
        self.end_token = end_token
        self.beam_size = beam_size
        self.max_length = max_length

        self._step = 0
        self.hypotheses  = []
        self.leaves = [{
            'path': tf.expand_dims(start_token, 0),
            'score': tf.expand_dims([tf.constant(0.)], 0)
            }]

    def add_node(self, leaf, node, score):
        return {
                'path': tf.concat([leaf['path'], node], axis=-1),
                'score': tf.concat([leaf['score'], tf.add(leaf['score'][:, -1], score)], axis=-1)
                }

    def add_hypothesis(self, hypothesis):
        self.hypotheses.append(hypothesis)

    def set_leaves(self, leaves):
        self.leaves = leaves

    def eval_path(self, inputs, path, training=False):
        # shape (batch_size, predicted_sequence_size, vocab_size)
        predictions = self.model(inputs=[*inputs, path], training=training)

        # select the last word from the seq_len dimension
        # shape (batch_size, 1, vocab_size)
        predictions = predictions[:, -1:, :]

        # select elements with top K probabilities
        top_args = tf.math.top_k(predictions, k=self.beam_size)

        predicted_ids = tf.cast(top_args.indices, tf.int32)  # shape (batch, 1, beam_size)
        predicted_scores = tf.cast(tf.map_fn(tf.math.log, top_args.values), tf.float32)  # shape (batch, 1, beam_size)

        return {'path': predicted_ids, 'score': predicted_scores}

    def choose_leaves(self):
        return sorted(
                self.leaves,
                key=lambda x: x['score'][:, -1],
                reverse=True
                )[:self.beam_size]

    def step(self, inputs, max_hypothesis):

        leaves = []
        for leaf in self.choose_leaves():

            node = self.eval_path(inputs, leaf['path'])  # shape (batch, 1, beam_size)

            hypotheses = []
            for i in range(self.beam_size):

                hypothesis = self.add_node(leaf, node['path'][:, :, i], node['score'][:, :, i])

                if tf.equal(node['path'][:, :, i], self.end_token[0]) \
                or (self.max_length and hypothesis['path'].shape[-1] >= self.max_length):
                    self.add_hypothesis(hypothesis)
                    continue

                hypotheses.append(hypothesis)

            leaves.extend(hypotheses)

        self.set_leaves(leaves)
        self._step += 1

        return leaves

    def process_hypotheses(self, length_penalty):
        return [{
            'path': tf.squeeze(h['path'], axis=0),
            'score': tf.squeeze(tf.math.divide(
                h['score'][:, -1],
                tf.pow(h['score'].shape[-1], tf.constant(length_penalty))
                ), axis=0)
            } for h in self.hypotheses]

    def run(self, inputs, max_hypothesis=8, max_depth=32, length_penalty=0.7):

        while self.leaves and self._step <= max_depth:
            self.step(inputs, max_hypothesis)

        return sorted(self.process_hypotheses(length_penalty),
            key=lambda h: h['score'],
            reverse=True
            )[:max_hypothesis]
