import tensorflow as tf
import numpy as np

#cell = tf.keras.layers.SimpleRNNCell


@tf.custom_gradient
def calc_spikes(v_scaled):
    z = tf.greater(v_scaled, 0)
    z = tf.cast(z, dtype=tf.float32)

    def grad(dy):
        dE_dz = dy
        dz_dv_scaled = tf.maximum(1 - tf.abs(v_scaled), 0)

        dE_dv_scaled = dE_dz * dz_dv_scaled

        return dE_dv_scaled
    return z, grad


@tf.custom_gradient
def calc_spikes_fast_sigmoid_surrogate(v_scaled):
    z = tf.greater(v_scaled, 0)
    z = tf.cast(z, dtype=tf.float32)

    def grad(dy):
        dE_dz = dy
        dz_dv_scaled = tf.cast(1. / (1 + 100. * tf.abs(v_scaled))**2, dtype='float32')

        dE_dv_scaled = dE_dz * dz_dv_scaled

        return dE_dv_scaled
    print('crazy surrogate is used right now')
    return z, grad


# @tf.keras.utils.register_keras_serializable(package='CustomNeurons', name='IntegratorCell')
class IntegratorCell(tf.keras.layers.Layer):
    """
    Simple layer that integrates the outputs of previous layer
    """

    def __init__(self, n_in, n_rec, fixed, w_regu=0., **kwargs):
        super(IntegratorCell, self).__init__(**kwargs)
        self.n_in = n_in
        self.n_rec = n_rec
        self.fixed = fixed
        self.w_regu = w_regu

        # w_in = tf.random.normal([n_in, n_rec]) / np.sqrt(n_in)
        # self.w_in = tf.Variable(initial_value=w_in, name="InputWeight", dtype=tf.float32, trainable=True)

    @property
    def state_size(self):
        """
        Not implemented yet.
        Returns the state size of the neurons
        :return:
        """
        return self.n_rec, self.n_rec

    def build(self, input_shape):
        self.w_in = self.add_weight('InputWeights', shape=(self.n_in, self.n_rec),
                                    initializer='random_normal',
                                    trainable=not self.fixed,
                                    regularizer=tf.keras.regularizers.L1(l1=self.w_regu),
                                    )

    def __call__(self, inputs, state, training=False):
        z, v = state

        i_in = tf.matmul(inputs, self.w_in)
        new_v = v + i_in
        new_z = tf.nn.softmax(new_v)

        state = [new_z, new_v]

        return new_z, state

    def get_config(self):
        # required to save the model. Add all variables that are present in __init__
        config = super().get_config()
        config.update({'n_in': self.n_in,
                       'n_rec': self.n_rec})
        return config


# @tf.keras.utils.register_keras_serializable(package='Custom', name='StdLeakLIF')
class StdLeakLIF(IntegratorCell):
    def __init__(self, n_in, n_rec, tau, thr, dt,
                 recurrence=False, fixed=False, weight_mu=0, weight_std=0.01, w_regu=0., **kwargs):
        super(StdLeakLIF, self).__init__(n_in=n_in, n_rec=n_rec, fixed=fixed, **kwargs)
        self.n_in = n_in
        self.n_rec = n_rec
        self.tau = tau
        self.dt = dt
        self.thr = thr
        self.recurrence = recurrence
        self.fixed = fixed
        self.weight_mu = weight_mu
        self.weight_std = weight_std
        self.w_regu = w_regu

        # w_rec = tf.zeros([n_rec, n_rec])
        # self.w_rec = tf.Variable(initial_value=w_rec, name="RecurrentWeight", dtype=tf.float32, trainable=recurrence)

    def compute_z(self, v):
        # z = calc_spikes((v - self.thr) / self.thr)
        z = calc_spikes_fast_sigmoid_surrogate((v - self.thr) / self.thr)
        return z

    def build(self, input_shape):
        self.w_in = self.add_weight('InputWeights', shape=(self.n_in, self.n_rec),
                                    initializer=tf.keras.initializers.RandomNormal(mean=self.weight_mu,
                                                                                   stddev=self.weight_std),
                                    trainable=not self.fixed,
                                    regularizer=tf.keras.regularizers.L1(l1=self.w_regu)
                                    )
        self.w_rec = self.add_weight('RecurrentWeights', shape=(self.n_rec, self.n_rec),
                                     initializer='zeros',
                                     trainable=self.recurrence,
                                     regularizer=tf.keras.regularizers.L1(l1=self.w_regu),
                                     )

    @property
    def state_size(self):
        """
        Not implemented yet.
        Returns the state size of the neurons
        :return:
        """
        return self.n_rec, self.n_rec, self.n_rec

    def __call__(self, inputs, state, training=False):
        z, v, t = state

        i_in = tf.matmul(inputs, self.w_in) + tf.matmul(z, self.w_rec)

        h = tf.cast((i_in != 0), dtype='float32')

        new_t = t + tf.cast((h != 1.0), dtype='float32')

        new_v = v * (1 - z)

        low_pass = tf.cast(tf.greater(v, -1), dtype=tf.float32)
        new_v += -(1. - low_pass) * (v + 1.)  # sets v<-1 to v=-1

        new_v = new_v * (1 - (self.dt / self.tau)) ** (h * (new_t + 1))  # new dt

        new_t = new_t * tf.cast((h != 1.0), dtype='float32')

        new_v = new_v + i_in

        new_z = self.compute_z(new_v)

        state = [new_z, new_v, new_t]

        return [new_z, new_v], state

    def get_config(self):
        # required to save the model. Add all variables that are present in __init__ (without those of parent layer)
        config = super().get_config()
        config.update({'n_rec': self.n_rec,
                       'tau': self.tau,
                       'dt': self.dt,
                       'thr': self.thr,
                       'recurrence': self.recurrence,
                       'fixed': self.fixed})
        return config


class StdLeakyIntegrator(IntegratorCell):
    def __init__(self, n_in, n_rec, tau, dt, recurrence=False, fixed=False, w_regu=0., **kwargs):
        super(StdLeakyIntegrator, self).__init__(n_in=n_in, n_rec=n_rec, fixed=fixed, **kwargs)
        self.n_in = n_in
        self.n_rec = n_rec
        self.tau = tau
        self.dt = dt
        self.recurrence = recurrence
        self.fixed = fixed
        self.w_regu = w_regu

        # w_rec = tf.zeros([n_rec, n_rec])
        # self.w_rec = tf.Variable(initial_value=w_rec, name="RecurrentWeight", dtype=tf.float32, trainable=recurrence)

    def build(self, input_shape):
        self.w_in = self.add_weight('InputWeights', shape=(self.n_in, self.n_rec),
                                    initializer=tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.01),
                                    trainable=not self.fixed,
                                    regularizer=tf.keras.regularizers.L1(l1=self.w_regu),
                                    )
        self.w_rec = self.add_weight('RecurrentWeights', shape=(self.n_rec, self.n_rec),
                                     initializer='zeros',
                                     trainable=self.recurrence,
                                     regularizer=tf.keras.regularizers.L1(l1=self.w_regu),
                                     )

    @property
    def state_size(self):
        """
        Not implemented yet.
        Returns the state size of the neurons
        :return:
        """
        return self.n_rec, self.n_rec, self.n_rec

    def __call__(self, inputs, state, training=False):
        z, v, t = state

        i_in = tf.matmul(inputs, self.w_in) + tf.matmul(z, self.w_rec)

        h = tf.cast((i_in != 0), dtype='float32')

        new_t = t + tf.cast((h != 1.0), dtype='float32')

        low_pass = tf.cast(tf.greater(v, -1), dtype=tf.float32)
        new_v = v - (1. - low_pass) * (v + 1.)

        new_v = new_v * (1 - (self.dt / self.tau)) ** (h * (new_t + 1))  # new dt

        new_t = new_t * tf.cast((h != 1.0), dtype='float32')

        new_v = new_v + i_in

        new_z = tf.nn.softmax(new_v)

        state = [new_z, new_v, new_t]

        return new_z, state

    def get_config(self):
        # required to save the model. Add all variables that are present in __init__ (without those of parent layer)
        config = super().get_config()
        config.update({'n_rec': self.n_rec,
                       'tau': self.tau,
                       'dt': self.dt,
                       'recurrence': self.recurrence,
                       'fixed': self.fixed})
        return config

