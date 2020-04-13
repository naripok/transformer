import tensorflow as tf


def loss_function(max_length=32):
    def _loss_function(y_true, y_pred):
        y_true = tf.reshape(y_true, shape=(-1, max_length - 1))

        loss = tf.keras.losses.SparseCategoricalCrossentropy(
            from_logits=False, reduction='none')(y_true, y_pred)

        mask = tf.cast(tf.not_equal(y_true, 0), tf.float32)
        loss = tf.multiply(loss, mask)

        return tf.reduce_mean(loss)
    return _loss_function


class CustomSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
    def __init__(self, d_model, warmup_steps=4000):
        super(CustomSchedule, self).__init__()
        self.d_model = d_model
        self.warmup_steps = warmup_steps
        self.d_model_t = tf.cast(self.d_model, tf.float32)

    def __call__(self, step):
        arg1 = tf.math.rsqrt(step)
        arg2 = step * (self.warmup_steps**-1.5)

        return tf.math.rsqrt(self.d_model_t) * tf.math.minimum(arg1, arg2)

    def get_config(self):
        return {
            'd_model': self.d_model,
            'warmup_steps': self.warmup_steps
        }


def accuracy(max_length=32):
    def _accuracy(y_true, y_pred):
        # ensure labels have shape (batch_size, max_length - 1)
        y_true = tf.reshape(y_true, shape=(-1, max_length - 1))
        return tf.keras.metrics.sparse_categorical_accuracy(y_true, y_pred)
    return _accuracy
