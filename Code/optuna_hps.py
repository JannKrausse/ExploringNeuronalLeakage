import os
import joblib
import argparse
from distutils.util import strtobool

from gpu_selection import select_gpu
select_gpu(1)

import tensorflow as tf
import optuna
import numpy as np

from data import get_tonic_data
from model import create_model_optuna, compile_model
from train import train_model
from auxiliary import plot_study_results, save_study_results
from pruning import prune_model


def optuna_train_model(kwargs, dataset, train_args, callbacks, lr, batch_size, sparsity, es_patience):

    # dataset = get_data(dataset_name=dataset_name, nb_bins=50, spat_ds_fac=1.0)
    in_shape = (dataset['x_train_set'][0].shape[0], dataset['x_train_set'][0].shape[1])
    # in_shape = (dataset['train_set'].element_spec[0].shape[0], dataset['train_set'].element_spec[0].shape[1])
    out_shape = tf.unique(dataset['y_valid_set']).y.shape[0]

    train_args['fixed'] = False
    train_args['input_shape'] = in_shape
    train_args['output_shape'] = out_shape

    model = create_model_optuna(**train_args, kwargs=kwargs)

    sparsity_regularization_factor = 0.0
    model = compile_model(model, lr=lr, eagerly=False, sparsity_regularization_factor=sparsity_regularization_factor)

    model = train_model(model, dataset, batch_size=batch_size, epochs=1500, es_patience=es_patience,
                        callbacks=callbacks)

    pruned_model = prune_model(model, dataset, sparsity=sparsity, train_args=train_args, kwargs=kwargs)

    hist = pruned_model.history.history
    best_acc = max(hist['val_output_sparse_categorical_accuracy'])

    return best_acc


class Objective:
    def __init__(self, nb_models, dataset, train_args, lr, batch_size, sparsity, es_patience):
        os.environ['TF_GPU_ALLOCATOR'] = "cuda_malloc_async"
        self.nb_models = nb_models
        self.dataset = dataset
        self.train_args = train_args
        self.lr = lr
        self.batch_size = batch_size
        self.sparsity = sparsity
        self.es_patience = es_patience

    def __call__(self, trial):
        kwargs = dict()
        if self.train_args['neuron_type'] == 'StdLeakLIF':
            kwargs["tau"] = trial.suggest_float('tau', 1, 2000, log=True)
        elif self.train_args['neuron_type'] == 'LogLeakLIF':
            kwargs["a_log"] = trial.suggest_float('a_log', 0.01, 20., log=True)
            # kwargs["b_log"] = trial.suggest_float('b_log', 1, 1, log=True)
            kwargs["c_log"] = trial.suggest_float('c_log', 0.01, 0.5, log=True)
        elif self.train_args['neuron_type'] == 'InvLeakLIF':
            kwargs["a_inv"] = trial.suggest_float('a_inv', 0.0005, 0.5, log=True)
            kwargs["b_inv"] = trial.suggest_float('b_inv', 2., 20., log=True)
        elif self.train_args['neuron_type'] == 'PowLeakLIF':
            kwargs["a_pow"] = trial.suggest_float('a_root', 1., 2000., log=True)
            kwargs["b_pow"] = trial.suggest_int('b_root', 1., 10., log=True)
        elif self.train_args['neuron_type'] == 'LinLeakLIF':
            kwargs["base_leak"] = trial.suggest_float('base_leak', 5e-12, 1.0, log=True)
        elif self.train_args['neuron_type'] == 'RootLeakLIF':
            kwargs["a_root"] = trial.suggest_float('a_root', 5e-6, 0.5, log=True)
            kwargs["b_root"] = trial.suggest_float('b_root', 0.05, 0.5, log=True)
        else:
            raise Exception(f'Neuron model type {self.train_args["neuron_type"]} not specified.')
        # kwargs["lr"] = trial.suggest_float('lr', 1e-5, 1e-2, log=True)

        self.callbacks = [
            # TFKerasPruningCallback(trial, 'val_loss')  # Optuna's pruner considering trials
        ]

        accs = np.array([])
        for i in range(self.nb_models):
            best_acc = optuna_train_model(kwargs, self.dataset, self.train_args, self.callbacks, self.lr,
                                          self.batch_size, self.sparsity, self.es_patience)
            accs = np.append(accs, best_acc)

        return accs.mean()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_name', default='NMNIST', type=str)
    parser.add_argument('--neuron_type', default='Std', type=str)
    parser.add_argument('--recurrence', dest='recurrence', type=lambda x: bool(strtobool(x)), default=False)
    parser.add_argument('--leaky_output', dest='leaky_output', type=lambda x: bool(strtobool(x)), default=False)
    parser.add_argument('--nb_time_bins', default=50, type=int)
    parser.add_argument('--load_existing_study', dest='load_existing_study', type=lambda x: bool(strtobool(x)),
                        default=False)
    parser.add_argument('--hidden_units', default=0, type=int)
    parser.add_argument('--es_patience', default=15, type=int)
    parser.add_argument('--nb_models_per_trial', default=5, type=int)
    parser.add_argument('--nb_trials', default=50, type=int)
    args = parser.parse_args()

    train_args = dict()

    dataset_name = args.dataset_name

    if dataset_name == 'NMNIST':
        spat_ds = 0.5
        nb_models = 1
        train_args['hidden_units'] = 50
        batch_size = 8192
        lr = 1e-3
        train_args['num_hidden_layers'] = 1
        sparsity = 0.55
    elif dataset_name == 'SHD':
        spat_ds = 1.0
        nb_models = args.nb_models_per_trial
        train_args['hidden_units'] = 100
        batch_size = 2048
        lr = 1e-3
        train_args['num_hidden_layers'] = 1
        sparsity = 0.65
    elif dataset_name == 'DVSGesture':
        spat_ds = 0.25
        nb_models = args.nb_models_per_trial
        train_args['hidden_units'] = 50
        batch_size = 2028
        lr = 3e-4
        train_args['num_hidden_layers'] = 2
        sparsity = 0.8

    train_args['neuron_type'] = args.neuron_type + 'LeakLIF'
    train_args['dropout'] = 0.4
    train_args['recurrence'] = args.recurrence
    train_args['leaky_output'] = args.leaky_output
    nb_bins = args.nb_time_bins

    if args.hidden_units > 0:
        train_args['hidden_units'] = args.hidden_units

    dataset = get_tonic_data(dataset_name=dataset_name, nb_bins=nb_bins, spat_ds_fac=spat_ds)

    study_name = f"{dataset_name}-{nb_bins}-{spat_ds}_{train_args['neuron_type']}_recurrence={args.recurrence}" \
                 f"_leakyoutput={args.leaky_output}_{train_args['hidden_units']}neurons_" \
                 f"{int(100*train_args['dropout'])}dropout_{nb_models}modelsmean_nohyperband"

    load_existing_study = args.load_existing_study
    n_trials_per_cycle = 1
    n_cycles = int(args.nb_trials / n_trials_per_cycle)  # total number of trials is n_trials_per_save*n_cycles
    if not load_existing_study:
        lg = 0
    else:
        lg = len(joblib.load(f'/home/kiasic/repos/jann/LeakPaper/CurrentOptunaStudyDump/{study_name}.pkl').trials)
    for i in range(n_cycles - int(lg / n_trials_per_cycle)):
        if i == 0 and load_existing_study is False:
            study = optuna.create_study(direction='maximize')  # , pruner=optuna.pruners.HyperbandPruner())
        else:
            study = joblib.load(f'/home/kiasic/repos/jann/LeakPaper/CurrentOptunaStudyDump/{study_name}.pkl')
        objective = Objective(nb_models=nb_models, dataset=dataset, train_args=train_args, lr=lr,
                              batch_size=batch_size, sparsity=sparsity, es_patience=args.es_patience)
        study.optimize(objective, n_trials=n_trials_per_cycle, gc_after_trial=True)
        joblib.dump(study, f'/home/kiasic/repos/jann/LeakPaper/CurrentOptunaStudyDump/{study_name}.pkl')

    save_study_results(study, study_name)
    plot_study_results(study, study_name)

    print('Optuna study finished.')


if __name__ == "__main__":
    main()
