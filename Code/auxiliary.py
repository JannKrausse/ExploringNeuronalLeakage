import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import optuna.trial
import tensorflow as tf

tf_seed = 53
# tf_seed = np.random.randint(1, 10000)
tf.random.set_seed(tf_seed)


@tf.function
def spike_share(y, spikes):
    # y not used
    size = tf.size(spikes)  # spike share over whole batch
    # size = spikes.shape[0]  # total number of spikes in one sample
    return tf.reduce_sum(spikes) / tf.cast(size, tf.float32)


@tf.function
def custom_lr_schedule(epoch, lr):
    lr_ini = 0.0001
    if epoch < 100:
        return lr_ini
    elif epoch < 200:
        return lr_ini/10.
    elif epoch < 1000:
        return lr_ini/100.
    else:
        return lr_ini/1000.


def save_study_results(study, study_name):
    current_dir = Path(__file__).parent
    save_study_dir_name = f"optuna_results/{study_name}"
    save_study_path = current_dir / save_study_dir_name

    if not os.path.isdir(save_study_path):
        os.mkdir(save_study_path)

    if not os.path.isfile(f"{save_study_path}/results.txt"):
        np.savetxt(f"{save_study_path}/results.txt", [])

    results = np.loadtxt(f"{save_study_path}/results.txt")

    results = np.append(results, f"Best trial number: {study.best_trial.number}")
    results = np.append(results, f" Acc: {study.best_trial.value}")
    for k, v in study.best_params.items():
        results = np.append(results, f"  {k}: {v}")
    results = np.append(results, "------------------")
    for i, t in enumerate(study.trials):
        results = np.append(results, f"Trial number: {i}")
        results = np.append(results, f" Acc: {t.value}")
        for k, v in t.params.items():
            results = np.append(results, f"  {k}: {v}")

    np.savetxt(f"{save_study_path}/results.txt", results, fmt='%s')


def plot_study_results(study, study_name):
    current_dir = Path(__file__).parent
    save_study_dir_name = f"optuna_results/{study_name}"
    save_study_path = current_dir / save_study_dir_name

    param_values_dict = dict()
    acc_values = np.array([])
    l = len(study.trials[0].params.keys())
    for i in range(l):
        param_values_dict[i] = np.array([])
    for t in study.trials:
        if t.state == optuna.trial.TrialState.COMPLETE:
            for i, v in enumerate(t.params.values()):
                param_values_dict[i] = np.append(param_values_dict[i], v)
            acc_values = np.append(acc_values, t.value)

    np.savetxt(save_study_path/"_acc", acc_values)
    for i in range(l):
        np.savetxt(save_study_path / f"_{list(study.trials[0].params.keys())[i]}", param_values_dict[i])

    for i in range(l):
        plt.plot(param_values_dict[i], acc_values, ls='', marker='.')
        # plt.text(x=param_values_dict[i][0], y=acc_values[0], s=f'{list(study.trials[0].params.keys())[i]}')
        plt.xlabel(f'{list(study.trials[0].params.keys())[i]}')
        plt.ylabel(f'accuracy')
        plt.xscale('log')
        plt.savefig(save_study_path/f"results_plot_{list(study.trials[0].params.keys())[i]}")
        plt.clf()
