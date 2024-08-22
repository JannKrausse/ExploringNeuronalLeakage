import os
import argparse
from distutils.util import strtobool

import numpy as np
from pathlib import Path

from data import get_tonic_data
from model import compile_model, save_model, create_model_optuna
from pruning import prune_model
from train import train_model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_name', default='NMNIST', type=str)
    parser.add_argument('--neuron_type', default='Std', type=str)
    parser.add_argument('--spike_regu', default=0., type=float)
    parser.add_argument('--nb_time_bins', default=50, type=int)
    parser.add_argument('--recurrence', dest='recurrence', type=lambda x: bool(strtobool(x)), default=False)
    parser.add_argument('--hidden_units', default=50, type=int)
    parser.add_argument('--lr', default=1e-3, type=float)
    parser.add_argument('--es_patience', default=15, type=int)
    parser.add_argument('--epochs', default=10000, type=int)
    parser.add_argument('--test_run', dest='test_run', type=lambda x: bool(strtobool(x)), default=False)
    parser.add_argument('--leaky_output', dest='leaky_output', type=lambda x: bool(strtobool(x)), default=False)
    parser.add_argument('--custom_name', default='', type=str)
    parser.add_argument('--pruning', dest='pruning', type=lambda x: bool(strtobool(x)), default=True)
    args = parser.parse_args()

    # prohibits tensorflow from claiming all available GPU memory
    from gpu_selection import select_gpu
    select_gpu(0)
    import tensorflow as tf

    data_args = dict()
    train_args = dict()
    kwargs = dict()  # neuron params
    data_args['dataset_name'] = args.dataset_name
    data_args['nb_bins'] = args.nb_time_bins
    callbacks = []

    # set parameters of dataset, network architecture and connection sparsity, and neuron parameters here depending
    # on the used dataset
    data_args['spat_ds'] = 0.5
    train_args['num_hidden_layers'] = 1
    train_args['hidden_units'] = 50
    sparsity = 0.55
    lr = 1e-3
    batch_size = 8192
    if args.neuron_type == 'Std':
        kwargs['tau'] = 2.
    elif args.neuron_type == 'Lin':
        kwargs["base_leak"] = 1 / 4.
    elif args.neuron_type == 'Non':
        pass
    elif args.neuron_type == 'Log':
        kwargs["a_log"] = 1 / 4.
        kwargs["c_log"] = 1 / 4.
    elif args.neuron_type == 'Pow':
        kwargs["a_pow"] = 256.
        kwargs["b_pow"] = 7
    elif args.neuron_type == 'Root':
        kwargs["a_root"] = 1 / 128.
        kwargs["b_root"] = 0.4045
    elif args.neuron_type == 'Inv':
        kwargs["a_inv"] = 1 / 8.
        kwargs["b_inv"] = 10.44

    model_save = True
    if args.test_run:
        train_args['hidden_units'] = args.hidden_units
        lr = args.lr
        model_save = False

    dataset = get_tonic_data(dataset_name=data_args['dataset_name'], nb_bins=data_args['nb_bins'],
                             spat_ds_fac=data_args['spat_ds'])

    in_shape = (dataset['x_train_set'][0].shape[0], dataset['x_train_set'][0].shape[1])
    out_shape = tf.unique(dataset['y_valid_set']).y.shape[0]

    train_args['fixed'] = False
    train_args['input_shape'] = in_shape
    train_args['output_shape'] = out_shape
    train_args['dropout'] = 0.4
    train_args['recurrence'] = args.recurrence
    train_args['leaky_output'] = args.leaky_output

    train_args['neuron_type'] = f"{args.neuron_type}LeakLIF"

    model = create_model_optuna(**train_args, kwargs=kwargs)

    sparsity_regularization_factor = args.spike_regu
    model = compile_model(model, lr=lr, eagerly=False, sparsity_regularization_factor=sparsity_regularization_factor)

    model = train_model(model, dataset, batch_size=batch_size, epochs=args.epochs, es_patience=args.es_patience,
                        callbacks=callbacks)

    if args.pruning:
        pruned_model = prune_model(model, dataset, sparsity=sparsity, train_args=train_args, kwargs=kwargs,
                                   spike_regu=args.spike_regu)

        hist = pruned_model.history.history
    else:
        hist = model.history.history
    best_acc = max(hist['val_output_sparse_categorical_accuracy'])

    save_path = f'{data_args["dataset_name"]}_{train_args["neuron_type"]}'

    # save model
    if model_save:
        save_model(model, saving_path=f"saved_models/{save_path}")

    # log model accuracy
    result_path = Path(__file__).parent / f"results_{args.custom_name}"
    if not os.path.isdir(result_path):
        os.mkdir(result_path)
    result_path = result_path / f'{save_path}'
    if not os.path.isdir(result_path):
        os.mkdir(result_path)
    if not os.path.isfile(f"{result_path}/acc.txt"):
        np.savetxt(f"{result_path}/acc.txt", [])
    run_results_acc = np.loadtxt(f"{result_path}/acc.txt")
    run_results_acc = np.append(run_results_acc, best_acc)
    np.savetxt(f"{result_path}/acc.txt", run_results_acc)


if __name__ == "__main__":
    main()
