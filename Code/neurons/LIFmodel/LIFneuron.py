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


# @tf.keras.utils.register_keras_serializable(package='CustomNeurons', name='IntegratorCell')
class IntegratorCell(tf.keras.layers.Layer):
    """
    Simple layer that integrates the outputs of previous layer
    """

    def __init__(self, n_in, n_rec, fixed, **kwargs):
        super(IntegratorCell, self).__init__(**kwargs)
        self.n_in = n_in
        self.n_rec = n_rec
        self.fixed = fixed

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
        self.w_in = self.add_weight('InputWeights', shape=(self.n_in, self.n_rec), initializer='random_normal',
                                    trainable=not self.fixed)

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


class LIF(IntegratorCell):
    def __init__(self, n_in, n_rec, tau, thr, dt, t_refrac=0, recurrence=False, decay_train_mode=0, fixed=False,
                 **kwargs):
        super(LIF, self).__init__(n_in=n_in, n_rec=n_rec, fixed=fixed, **kwargs)
        self.n_in = n_in
        self.n_rec = n_rec
        self.tau = tau
        self.dt = dt
        self.thr = thr
        self.t_refrac = t_refrac
        self.recurrence = recurrence
        self.decay_train_mode = decay_train_mode
        self.fixed = fixed

        # w_rec = tf.zeros([n_rec, n_rec])
        # self.w_rec = tf.Variable(initial_value=w_rec, name="RecurrentWeight", dtype=tf.float32, trainable=recurrence)

    def compute_z(self, v):
        z = calc_spikes((v - self.thr) / self.thr)
        return z

    def build(self, input_shape):
        if not self.fixed:
            self.w_in = self.add_weight('InputWeights', shape=(self.n_in, self.n_rec), initializer='random_normal',
                                        trainable=True)
            self.w_rec = self.add_weight('RecurrentWeights', shape=(self.n_rec, self.n_rec), initializer='zeros',
                                         trainable=self.recurrence)
        else:
            # diagonal input for current injection
            assert self.n_in == self.n_rec
            self.w_in = self.add_weight('InputWeights', shape=(self.n_in, self.n_rec), initializer='Identity',
                                        trainable=False)
            self.w_rec = self.add_weight('RecurrentWeights', shape=(self.n_rec, self.n_rec), initializer='zeros',
                                         trainable=False)
        decay = np.exp(-self.dt / self.tau)
        if self.decay_train_mode == 0:
            self.decay = tf.Variable(initial_value=decay, trainable=False, dtype=tf.float32, name='decay')
        elif self.decay_train_mode == 1:
            self.decay = tf.Variable(initial_value=decay, trainable=True, dtype=tf.float32, name='decay',
                                     constraint=lambda t: tf.clip_by_value(t, 10**(-9), 1-10**(-4)))
        elif self.decay_train_mode == 2:
            self.decay = tf.Variable(initial_value=decay*tf.ones(self.n_rec), trainable=True, dtype=tf.float32,
                                     name='decay', constraint=lambda t: tf.clip_by_value(t, 10**(-9), 1-10**(-4)))
        else:
            print('Wrong decay train mode specified')

    @property
    def state_size(self):
        """
        Not implemented yet.
        Returns the state size of the neurons
        :return:
        """
        return self.n_rec, self.n_rec, self.n_rec

    def __call__(self, inputs, state, training=False):
        z, v, r = state

        i_in = tf.matmul(inputs, self.w_in) + tf.matmul(z, self.w_rec)
        i_reset = z * self.thr

        new_v = self.decay * v + i_in - i_reset

        if self.t_refrac:
            is_refrac = tf.greater_equal(r, 1.)
            refrac_mask = tf.zeros_like(z)
            new_z = tf.where(is_refrac, refrac_mask, self.compute_z(new_v))
            new_r = tf.stop_gradient(tf.clip_by_value(r + self.t_refrac * z - 1, 0, float(self.t_refrac)))
        else:
            new_z = self.compute_z(new_v)
            new_r = tf.stop_gradient(tf.zeros_like(r))

        state = [new_z, new_v, new_r]

        return [new_z, new_v], state

    def get_config(self):
        # required to save the model. Add all variables that are present in __init__ (without those of parent layer)
        config = super().get_config()
        config.update({'n_rec': self.n_rec,
                       'tau': self.tau,
                       'dt': self.dt,
                       'thr': self.thr,
                       't_refrac': self.t_refrac,
                       'recurrence': self.recurrence,
                       'decay_train_mode': self.decay_train_mode,
                       'fixed': self.fixed})
        return config


class ALIF(LIF):
    def __init__(self, n_in, n_rec, tau, thr, dt, tau_adap, t_refrac=0, recurrence=False, decay_train_mode=0,
                 fixed=False, adap_decay_train_mode=0, **kwargs):
        super(ALIF, self).__init__(n_in=n_in, n_rec=n_rec, tau=tau, thr=thr, dt=dt, t_refrac=t_refrac,
                                   recurrence=recurrence, decay_train_mode=decay_train_mode, fixed=fixed, **kwargs)
        self.tau_adap = tau_adap
        self.adap_decay_train_mode = adap_decay_train_mode

        if np.isscalar(thr):
            self.thr = np.ones(n_rec) * thr

    @property
    def state_size(self):
        """
        Not implemented yet.
        Returns the state size of the neurons
        :return:
        """
        return self.n_rec, self.n_rec, self.n_rec, self.n_rec

    def build(self, input_shape):
        super(ALIF, self).build(input_shape)
        adap_decay = np.exp(-self.dt / self.tau_adap)
        if self.adap_decay_train_mode == 0:
            self.adap_decay = tf.Variable(initial_value=adap_decay, trainable=False, dtype=tf.float32, name='adap_decay')
        elif self.adap_decay_train_mode == 1:
            self.adap_decay = tf.Variable(initial_value=adap_decay, trainable=True, dtype=tf.float32, name='adap_decay',
                                     constraint=lambda t: tf.clip_by_value(t, 10**(-9), 1-10**(-4)))
        elif self.adap_decay_train_mode == 2:
            self.adap_decay = tf.Variable(initial_value=adap_decay*tf.ones(self.n_rec), trainable=True, dtype=tf.float32,
                                     name='adap_decay', constraint=lambda t: tf.clip_by_value(t, 10**(-9), 1-10**(-4)))
        else:
            print('Wrong adaptive decay train mode specified')

    def compute_z(self, v, a):
        new_th = self.thr + a
        v_scaled = (v - new_th) / new_th
        z = calc_spikes(v_scaled)
        return z

    def __call__(self, inputs, state, training=False):
        z, v, a, r = state

        i_in = tf.matmul(inputs, self.w_in) + tf.matmul(z, self.w_rec)
        i_reset = z * (self.thr + a)

        new_a = self.adap_decay * a + z
        new_v = self.decay * v + i_in - i_reset

        if self.t_refrac:
            is_refrac = tf.greater_equal(r, 1.)
            refrac_mask = tf.zeros_like(z)
            new_z = tf.where(is_refrac, refrac_mask, self.compute_z(new_v, new_a))
            new_r = tf.stop_gradient(tf.clip_by_value(r + self.t_refrac * z - 1, 0, float(self.t_refrac)))
        else:
            new_z = self.compute_z(new_v, new_a)
            new_r = tf.stop_gradient(tf.zeros_like(r))

        state = [new_z, new_v, new_a, new_r]
        return [new_z, new_v, new_a], state

    def get_config(self):
        # required to save the model. Add all variables that are present in __init__ (without those of parent layer)
        config = super().get_config()
        config.update({'tau_adap': self.tau_adap, 'adap_decay_train_mode': self.adap_decay_train_mode})
        return config
