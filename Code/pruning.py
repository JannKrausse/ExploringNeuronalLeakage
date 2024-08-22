import tempfile

import tensorflow as tf
import numpy as np
import tensorflow_model_optimization as tfmot

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


class MyPrunableRNN(tf.keras.layers.RNN, tfmot.sparsity.keras.PrunableLayer):
    def __init__(self, cell, return_sequences, name, recurrence):
        self.cell = cell
        self.return_sequences = return_sequences
        self.recurrence = recurrence
        super(MyPrunableRNN, self).__init__(cell=self.cell, return_sequences=self.return_sequences, name=name)

    def get_prunable_weights(self):
        """Return input weights as prunable weights."""
        prunable_weights = [self.weights[0]]
        if self.recurrence:
            prunable_weights.append(self.weights[1])
        return prunable_weights

    def get_config(self):
        config = super().get_config()
        config.update({
            'cell': self.cell,
            'return_sequences': self.return_sequences,
        })
        return config

    @classmethod
    def from_config(cls, config, custom_objects=None):
        from keras.layers import deserialize as deserialize_layer

        cell = deserialize_layer(
            config.pop("cell"), custom_objects=custom_objects
        )
        num_constants = config.pop("num_constants", 0)
        layer = cls(cell, **config)
        layer._num_constants = num_constants
        return layer


def create_prunable_model(neuron_type, input_shape, hidden_units, num_hidden_layers, output_shape, dropout, fixed,
                          pruning_params, kwargs, recurrence=False, leaky_output=False, prunable=True,
                          only_hidden=True):

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
            raise Exception("The neuron model is unknown or ill-defined.")

        if prunable:
            if not only_hidden:
                hidden_layers[i], state_v_dict[i] = MyPrunableRNN(spiking_neuron, return_sequences=True, name=f'HL{i}',
                                                                  recurrence=recurrence)(dropout_layers[i])
            else:
                hidden_layers[i], state_v_dict[i] = \
                    tfmot.sparsity.keras.prune_low_magnitude(MyPrunableRNN(spiking_neuron,
                                                        return_sequences=True, name=f'HL{i}', recurrence=recurrence),
                                                        **pruning_params)(dropout_layers[i])
        else:
            hidden_layers[i], state_v_dict[i] = tf.keras.layers.RNN(
                spiking_neuron,
                return_sequences=True, name=f'HL{i}')(dropout_layers[i])

    dropout_last = tf.keras.layers.Dropout(dropout)(hidden_layers[num_hidden_layers - 1])

    recurrence = False
    if not leaky_output or neuron_type == 'NonLeakLIF':
        output_neuron = IntegratorCell(n_in=hidden_units, n_rec=output_shape, fixed=fixed)
    else:
        if neuron_type == 'StdLeakLIF':
            output_neuron = StdLeakyIntegrator(n_in=hidden_units, n_rec=output_shape, tau=kwargs['tau'], dt=1,
                                        recurrence=recurrence)
        elif neuron_type == 'InvLeakLIF':
            output_neuron = InvLeakyIntegrator(n_in=hidden_units, n_rec=output_shape, a_inv=kwargs['a_inv'],
                                                  b_inv=kwargs['b_inv'], dt=1.0, recurrence=recurrence)
        elif neuron_type == 'LinLeakLIF':
            output_neuron = LinLeakyIntegrator(n_in=hidden_units, n_rec=output_shape, base_leak=kwargs['base_leak'],
                                                  dt=1.0, recurrence=recurrence)
        elif neuron_type == 'PowLeakLIF':
            output_neuron = RootLeakyIntegrator(n_in=hidden_units, n_rec=output_shape, a_root=kwargs['a_pow'],
                                         b_root=kwargs['b_pow'], dt=1.0, recurrence=recurrence)
        elif neuron_type == 'LogLeakLIF':
            output_neuron = LogLeakyIntegrator(n_in=hidden_units, n_rec=output_shape, a_log=kwargs['a_log'],
                                            b_log=1.0, c_log=kwargs['c_log'], dt=1.0, recurrence=recurrence)
        elif neuron_type == 'RootLeakLIF':
            output_neuron = InvLeakyIntegrator(n_in=hidden_units, n_rec=output_shape, a_inv=kwargs['a_root'],
                                                  b_inv=kwargs['b_root'], dt=1.0, recurrence=recurrence)
        else:
            raise Exception(f"The neuron model {neuron_type} is unknown or ill-defined.")

    if prunable and not only_hidden:
        output = MyPrunableRNN(output_neuron, return_sequences=False, name='output',
                               recurrence=recurrence)(dropout_last)
    else:
        output = tf.keras.layers.RNN(output_neuron,
                                     return_sequences=False,
                                     name='output')(dropout_last)

    model = tf.keras.models.Model(inputs=inputs, outputs=[output, *hidden_layers.values(), *state_v_dict.values()])

    return model


def compile_prunable_model(model, lr, eagerly, sparsity_regularization_factor, only_hidden):
    metrics = {}
    if only_hidden:
        metrics['output'] = "sparse_categorical_accuracy"
    else:
        metrics['prune_low_magnitude_output'] = "sparse_categorical_accuracy"
    for l in model.layers:
        if l.name[0:22] == 'prune_low_magnitude_HL':
            metrics[l.name] = spike_share

    if only_hidden:
        loss = {'output': tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False)}
    else:
        loss = {'prune_low_magnitude_output': tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False)}
    for l in model.layers:
        if l.name[0:22] == 'prune_low_magnitude_HL':
            loss[l.name] = SparsityRegularization(sparsity_regularization_factor)

    model.compile(
            loss=loss,
            optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
            metrics=metrics,
            run_eagerly=eagerly,
        )

    return model


def prune_model(model, data, sparsity, train_args, kwargs=None, pruning_type='magnitude', only_hidden=True,
                spike_regu=0.):
    if pruning_type == 'magnitude':
        pruning_method = tfmot.sparsity.keras.prune_low_magnitude
    else:
        raise Exception('Wrong pruning method specified.')

    batch_size = 2048
    epochs = 300
    lr = 1e-3
    pruning_params = {
        'pruning_schedule': tfmot.sparsity.keras.ConstantSparsity(target_sparsity=sparsity,
                                                                  begin_step=0,
                                                                  end_step=-1,
                                                                  frequency=2)
    }

    if not only_hidden:
        model_for_pruning = pruning_method(model, **pruning_params)
    else:
        prunable = True
        model_for_pruning = create_prunable_model(**train_args, pruning_params=pruning_params, kwargs=kwargs,
                                                  prunable=prunable, only_hidden=only_hidden)
        w1 = model.get_weights()
        w2 = model_for_pruning.get_weights()
        for i in range(train_args['num_hidden_layers']):
            w2[5*i] = w1[2*i]
            w2[5*i + 1] = w1[2*i + 1]
        w2[-1] = w1[-1]
        model_for_pruning.set_weights(w2)
    model_for_pruning = compile_prunable_model(model_for_pruning, lr=lr, eagerly=True,
                                               sparsity_regularization_factor=spike_regu, only_hidden=only_hidden)

    from train import train_model
    logdir = tempfile.mkdtemp()
    callbacks = [
        tfmot.sparsity.keras.UpdatePruningStep(),
        tfmot.sparsity.keras.PruningSummaries(log_dir=logdir),
    ]
    model_for_pruning = train_model(model_for_pruning, data=data, batch_size=batch_size, epochs=epochs, es_patience=10,
                                    callbacks=callbacks)

    return model_for_pruning
