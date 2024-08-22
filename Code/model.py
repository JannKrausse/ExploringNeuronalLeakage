import os

import tensorflow as tf
import numpy as np

from neurons.LIFmodel.LIFneuron import LIF, IntegratorCell
from neurons.SleepyLeakLIF.StdLeakLIF import StdLeakLIF, StdLeakyIntegrator
from neurons.SleepyLeakLIF.LogLeakLIF import LogLeakLIF, LogLeakyIntegrator
from neurons.SleepyLeakLIF.RootLeakLIF import RootLeakLIF, RootLeakyIntegrator
from neurons.SleepyLeakLIF.InvLeakLIF import InvLeakLIF, InvLeakyIntegrator
from neurons.SleepyLeakLIF.NonLeakLIF import NonLeakLIF
from neurons.SleepyLeakLIF.LinLeakLIF import LinLeakLIF, LinLeakyIntegrator
from auxiliary import spike_share
from sparsity_loss import SparsityRegularization

# tf_seed = 53
tf_seed = np.random.randint(1, 10000)
tf.random.set_seed(tf_seed)


def compile_model(model, lr, eagerly, sparsity_regularization_factor):
    metrics = dict()
    metrics['output'] = "sparse_categorical_accuracy"
    for l in model.layers:
        if l.name[0:2] == 'HL':
            metrics[l.name] = spike_share

    loss = {'output': tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False)}
    for l in model.layers:
        if l.name[0:2] == 'HL':
            loss[l.name] = SparsityRegularization(sparsity_regularization_factor)

    model.compile(
            loss=loss,
            optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
            metrics=metrics,
            run_eagerly=eagerly,
        )

    return model


def save_model(model, saving_path):
    model.save(saving_path)
    model.save(os.path.join(saving_path, "allinone.h5"))


def create_model_optuna(neuron_type, input_shape, hidden_units, num_hidden_layers, output_shape, dropout,
                        fixed, kwargs, recurrence=False, leaky_output=False):

    inputs = tf.keras.layers.Input(shape=input_shape)

    hidden_layers = {}
    dropout_layers = {}
    state_v_dict = {}
    for i in range(num_hidden_layers):
        if i == 0:
            dropout_layers[i] = tf.keras.layers.Dropout(dropout)(inputs)
            n_in = inputs.shape[-1]
        else:
            dropout_layers[i] = tf.keras.layers.Dropout(dropout)(hidden_layers[i - 1])
            n_in = hidden_units

        if neuron_type == 'StdLeakLIF':
            spiking_neuron = StdLeakLIF(n_in=n_in, n_rec=hidden_units, tau=kwargs['tau'], thr=1, dt=1,
                                        recurrence=recurrence)
        elif neuron_type == 'InvLeakLIF':
            spiking_neuron = InvLeakLIF(n_in=n_in, n_rec=hidden_units, a_inv=kwargs['a_inv'], b_inv=kwargs['b_inv'],
                                        thr=1.0, dt=1.0, recurrence=recurrence)
        elif neuron_type == 'LinLeakLIF':
            spiking_neuron = LinLeakLIF(n_in=n_in, n_rec=hidden_units, base_leak=kwargs['base_leak'], thr=1.0, dt=1.0,
                                        recurrence=recurrence)
        elif neuron_type == 'PowLeakLIF':
            spiking_neuron = RootLeakLIF(n_in=n_in, n_rec=hidden_units, a_root=kwargs['a_pow'],
                                         b_root=kwargs['b_pow'], thr=1.0, dt=1.0, recurrence=recurrence)
        elif neuron_type == 'LogLeakLIF':
            spiking_neuron = LogLeakLIF(n_in=n_in, n_rec=hidden_units, a_log=kwargs['a_log'], b_log=1.0,
                                        c_log=kwargs['c_log'], thr=1.0, dt=1.0, recurrence=recurrence)
        elif neuron_type == 'NonLeakLIF':
            spiking_neuron = NonLeakLIF(n_in=n_in, n_rec=hidden_units, thr=1.0, dt=1.0, recurrence=recurrence)
        elif neuron_type == 'RootLeakLIF':
            spiking_neuron = InvLeakLIF(n_in=n_in, n_rec=hidden_units, a_inv=kwargs['a_root'], b_inv=kwargs['b_root'],
                                        thr=1.0, dt=1.0, recurrence=recurrence)
        else:
            raise Exception(f"The neuron model {neuron_type} is unknown or ill-defined.")
        hidden_layers[i], state_v_dict[i] = tf.keras.layers.RNN(spiking_neuron, return_sequences=True,
                                                                name=f'HL{i}')(dropout_layers[i])

    dropout_last = tf.keras.layers.Dropout(dropout)(hidden_layers[num_hidden_layers - 1])

    recurrence = False
    if not leaky_output or neuron_type == 'NonLeakLIF':
        output = tf.keras.layers.RNN(IntegratorCell(n_in=hidden_units, n_rec=output_shape, fixed=fixed),
                                     return_sequences=False, name='output')(dropout_last)
    else:
        if neuron_type == 'StdLeakLIF':
            leaky_integrator = StdLeakyIntegrator(n_in=hidden_units, n_rec=output_shape, tau=kwargs['tau'], dt=1,
                                        recurrence=recurrence)
        elif neuron_type == 'InvLeakLIF':
            leaky_integrator = InvLeakyIntegrator(n_in=hidden_units, n_rec=output_shape, a_inv=kwargs['a_inv'],
                                                  b_inv=kwargs['b_inv'], dt=1.0, recurrence=recurrence)
        elif neuron_type == 'LinLeakLIF':
            leaky_integrator = LinLeakyIntegrator(n_in=hidden_units, n_rec=output_shape, base_leak=kwargs['base_leak'],
                                                  dt=1.0, recurrence=recurrence)
        elif neuron_type == 'PowLeakLIF':
            leaky_integrator = RootLeakyIntegrator(n_in=hidden_units, n_rec=output_shape, a_root=kwargs['a_pow'],
                                         b_root=kwargs['b_pow'], dt=1.0, recurrence=recurrence)
        elif neuron_type == 'LogLeakLIF':
            leaky_integrator = LogLeakyIntegrator(n_in=hidden_units, n_rec=output_shape, a_log=kwargs['a_log'],
                                            b_log=1.0, c_log=kwargs['c_log'], dt=1.0, recurrence=recurrence)
        elif neuron_type == 'RootLeakLIF':
            leaky_integrator = InvLeakyIntegrator(n_in=hidden_units, n_rec=output_shape, a_inv=kwargs['a_root'],
                                                  b_inv=kwargs['b_root'], dt=1.0, recurrence=recurrence)
        else:
            raise Exception(f"The neuron model {neuron_type} is unknown or ill-defined.")
        output = tf.keras.layers.RNN(leaky_integrator, return_sequences=False,
                                     name='output')(dropout_last)

    model = tf.keras.models.Model(inputs=inputs, outputs=[output, *hidden_layers.values(), *state_v_dict.values()])

    return model

