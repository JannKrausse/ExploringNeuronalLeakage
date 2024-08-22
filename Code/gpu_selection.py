from tensorflow import config
import os
from typing import Union


def select_gpu(gpu_ids: Union[int, list]):
    if isinstance(gpu_ids, int):
        os.environ["CUDA_VISIBLE_DEVICES"] = f"{gpu_ids}"
    elif isinstance(gpu_ids, list):
        cuda_devices = ""
        for gpu in gpu_ids:
            cuda_devices += str(gpu) + ", "
        os.environ["CUDA_VISIBLE_DEVICES"] = f"{cuda_devices}"
    else:
        raise TypeError('Please provide a single GPU ID or a list of IDs')
    gpus = config.list_physical_devices('GPU')
    for gpu in gpus:
        config.experimental.set_memory_growth(gpu, True)
