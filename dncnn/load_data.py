import logging
from functools import cache

import h5py
import numpy as np

logger = logging.getLogger(__name__)


@cache
def load_data(data_path) -> np.ndarray:
    logger.info("Loading data from: %s", data_path)
    data = h5py.File(data_path, "r")["30x30"]["layers"]
    data = np.array(data)
    logger.info("Loaded '%s' with size %s", data_path, str(data.shape))

    return data

if __name__ == "__main__":
    from pathlib import Path
    root_path = "../ILDCaloSim"
    train_noisy = "e-_Jun3/test/showers-10kE10GeV-RC10-1.hdf5"
    train_sharp = "e-_Jun3/test/showers-10kE10GeV-RC2-1.hdf5"
    data_path_noisy = Path(f"{root_path}/{train_noisy}")
    #  data_path_sharp = Path(f"{root_path}/{file_path_sharp}")
    #  noisy_data = load_data(train_noisy)
    #  print(noisy_data.sha

    import hdf5
    data_noisy = hdf5.load(data_path_noisy)["30x30"]["layers"]
    #  data_noisy = h5py.File(data_path_noisy, "r")["30x30"]["layers"]
    print(data_noisy.shape)
    
    print(data_noisy.shape)
