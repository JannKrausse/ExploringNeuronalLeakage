import tensorflow as tf


#  NEED TO FIGURE OUT HOW TO USE MULTIPLE INPUTS FOR SURROGATE GRADIENT FUNCTIONS
#  EITHER I DIDN'T CALL THEM CORRECTLY OR IT HAS OTHER PROBLEMS
@tf.custom_gradient
def calc_spikes_piecewise_linear_surrogate(v_scaled, beta):
    z = tf.greater(v_scaled, 0)
    z = tf.cast(z, dtype=tf.float32)

    def grad(dy):
        dE_dz = dy
        dz_dv_scaled = tf.maximum(1 - beta * tf.abs(v_scaled), 0)

        dE_dv_scaled = dE_dz * dz_dv_scaled

        return dE_dv_scaled
    return z, grad


class FastSigmoidSurrogate:
    def __init__(self, beta):
        self.beta = beta

    @tf.custom_gradient
    def __call__(self, v_scaled):
        z = tf.greater(v_scaled, 0)
        z = tf.cast(z, dtype=tf.float32)

        def grad(dy):
            dE_dz = dy
            dz_dv_scaled = 1. / (1 + self.beta * tf.abs(v_scaled)) ** 2

            dE_dv_scaled = dE_dz * dz_dv_scaled

            return dE_dv_scaled

        return z, grad


@tf.custom_gradient
def calc_spikes_fast_sigmoid_surrogate(v_scaled, beta):
    z = tf.greater(v_scaled, 0)
    z = tf.cast(z, dtype=tf.float32)

    def grad(dy):
        dE_dz = dy
        dz_dv_scaled = 1. / (1 + beta * tf.abs(v_scaled))**2

        dE_dv_scaled = dE_dz * dz_dv_scaled

        return dE_dv_scaled
    return z, grad
