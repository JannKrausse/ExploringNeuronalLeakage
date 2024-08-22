import tensorflow as tf


class SparsityRegularization(tf.keras.losses.Loss):
    def __init__(self, regularization_factor):
        self.regularization_factor = regularization_factor
        super().__init__()

    def call(self, y_true, y_pred):
        spike_count = tf.reduce_sum(y_pred)
        return self.regularization_factor * spike_count

    def get_config(self):
        config = super().get_config()
        config.update({
            'regularization_factor': self.regularization_factor,
        })
        return config

