from pathlib import Path

import numpy as np
import tonic
import tensorflow as tf


def get_tonic_data(dataset_name, nb_bins, spat_ds_fac):
    """Function that provides a Numpy array-based version of the Tonic datasets. dataset is the string defining the
    name of the respective dataset in the Tonic package. nb_bins and spat_ds_fac define the parameters of the used
    transforms provided by Tonic."""
    nb_bins = nb_bins
    spat_ds_fac = spat_ds_fac
    cache_dataset = Path(__file__).parent / f'data_cache/cache_{dataset_name}_{nb_bins}tbins_{spat_ds_fac}spatds.npz'

    if not cache_dataset.is_file():
        dataset = getattr(tonic.datasets, dataset_name)

        sensor_size = dataset.sensor_size
        sensor_size = (int(sensor_size[0]*spat_ds_fac), int(sensor_size[1]*spat_ds_fac), sensor_size[2])
        transform = tonic.transforms.Compose(
            [
                tonic.transforms.Downsample(spatial_factor=spat_ds_fac),
                tonic.transforms.ToFrame(sensor_size=sensor_size, n_time_bins=nb_bins),
            ]
        )

        sets = {}
        if dataset_name == 'NMNIST' or dataset_name == 'DVSGesture' or dataset_name == 'SHD' or dataset_name == 'DVSLip':
            sets['train_set'] = dataset('/snailta/Datasets/', transform=transform, train=True)
            sets['valid_set'] = dataset('/snailta/Datasets/', transform=transform, train=False)
        elif dataset_name == 'SSC':
            sets['train_set'] = dataset('/snailta/Datasets/', transform=transform, split='train')
            sets['valid_set'] = dataset('/snailta/Datasets/', transform=transform, split='valid')
            sets['test_set'] = dataset('/snailta/Datasets/', transform=transform, split='test')
        elif dataset_name == 'PokerDVS':
            sets['train_set'] = dataset('/snailta/Datasets/', transform=transform, train=True)
        elif dataset_name == 'ASLDVS':
            sets['train_set'] = dataset('/snailta/Datasets/', transform=transform)

        def data_generator(set):
            xshape = next(iter(set))[0].shape
            yshape = len(set)
            x_ = np.zeros((len(set), *xshape), dtype=np.int8)
            y_ = np.zeros(yshape)
            for i, (x, y) in enumerate(set):
                x = x.astype(np.uint8)
                x_[i] = x
                y_[i] = y
                print(f"\r{i+1}/{len(set)}", end='')
            x_ = x_.squeeze()
            y_ = np.asarray(y_, dtype=np.uint8).squeeze()
            return x_, y_

        data_dict = dict()
        for key, s in sets.items():
            x_, y_ = data_generator(s)

            if dataset_name != 'SHD':
                pos_x = x_[:, :, 0, :, :].reshape(*x_.shape[:2], -1)
                neg_x = x_[:, :, 1, :, :].reshape(*x_.shape[:2], -1)

                x_ = np.concatenate((pos_x, neg_x,), axis=2)
                del pos_x, neg_x

            names = f'x_{key}', f'y_{key}'
            data_dict[names[0]] = x_
            data_dict[names[1]] = y_
            # data_dict[key] = (x_, y_)
        np.savez(str(cache_dataset), **data_dict)

    else:
        data_dict = dict()
        with np.load(str(cache_dataset)) as load:
            for file in load.files:
                data_dict[file] = load[file]

    return data_dict


def make_tf_dataset(data_dict):
    tf_data_dict = dict()
    l = len(data_dict.keys())
    keys = list(data_dict.keys())
    values = list(data_dict.values())
    for i in range(int(l/2)):
        tf_data_dict[keys[2*i][2:]] = tf.data.Dataset.from_tensor_slices((values[2*i], values[2*i+1]))

    return tf_data_dict


def get_data(dataset_name, nb_bins, spat_ds_fac):
    dataset = get_tonic_data(dataset_name, nb_bins, spat_ds_fac)
    dataset = make_tf_dataset(dataset)

    return dataset

